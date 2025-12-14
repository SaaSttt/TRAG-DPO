import os
import torch
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM as LlavaForConditionalGeneration
import sys
sys.path.append('./clipmain')
from clip import CLIP
import gradio as gr
import torchvision.transforms as transforms
import numpy as np
from utils.calc_utils import calc_top1_labels

# ========= 配置 =========
MERGED_MODEL_PATH = "TRAG-DPO/llava-v1.5-7b-merged-test3"
SBERT_MODEL_PATH = "TRAG-DPO/paraphrase-MiniLM-L6-v2"
KNOWLEDGE_BASE_DIR = "TRAG-DPO/know"
FEATURES_PATH = "TRAG-DPO/LLaVA-1.1.3/clipmain/features_datasets.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(SBERT_MODEL_PATH, device=device)

# 工具函数
def paragraph_tokenize(text):
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def retrieve_relevant_context(question, knowledge_text, top_k=3):
    paragraphs = paragraph_tokenize(knowledge_text)
    if not paragraphs:
        return ""
    paragraph_embeddings = embedder.encode(paragraphs, convert_to_tensor=True, normalize_embeddings=True)
    query_embedding = embedder.encode(question, convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(query_embedding, paragraph_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(paragraphs)))
    return "\n\n".join([paragraphs[i] for i in top_results.indices])

def build_prompt(image_tag, context, question):
    return f"""USER:
This is an image of a tomato crop,
Relevant background knowledge is given below:
{context}
Now, you are an agricultural expert. Please answer the following question strictly based on the knowledge above.
Question: {question}
ASSISTANT:"""

def load_model():
    model = LlavaForConditionalGeneration.from_pretrained(
        MERGED_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(MERGED_MODEL_PATH, use_fast=False, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, use_fast=False, trust_remote_code=True)
    return model, processor, tokenizer

def get_knowledge_path(label):
    filename = f"{label}_new.txt"
    return os.path.join(KNOWLEDGE_BASE_DIR, filename)

def generate_answer(model, processor, tokenizer, image_path, question, context):
    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt("<image>", context, question)
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding="longest"
    ).to(device)

    generate_kwargs = {
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("ASSISTANT:")[-1].strip()

features_dict = np.load(FEATURES_PATH, allow_pickle=True).item()
ida_features_np = features_dict['ida_features']
tda_features_np = features_dict['tda_features']
lda_features_np = features_dict['lda_features']

retrieval_labels = torch.tensor(lda_features_np, dtype=torch.float32)
retrieval_txt = torch.sign(torch.tensor(tda_features_np, dtype=torch.float32))

model, processor, tokenizer = load_model()
modelclip = CLIP()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_knowledge_from_image(image):
    image_tensor = transform(image).unsqueeze(0).to(device).to(torch.float32)
    images_feature = modelclip.detect_image_for_eval(image_tensor, texts=None)
    ida_features_np = images_feature[0].cpu().numpy()
    query_img = torch.sign(torch.tensor(ida_features_np, dtype=torch.float32))
    query_labels = torch.tensor([[1] + [0] * 9], dtype=torch.float32)

    mAPi2t = calc_top1_labels(query_img, retrieval_txt, query_labels, retrieval_labels)
    y_pred = np.array([label.cpu().numpy().squeeze() for label in mAPi2t])
    pred_label = int(np.argmax(y_pred, axis=1)[0])

    knowledge_path = get_knowledge_path(pred_label)
    if not os.path.exists(knowledge_path):
        knowledge_text = ""
        first_line = "[无知识]"
    else:
        with open(knowledge_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            knowledge_text = f.read()

    print(f"===== 上传图片初始化 =====")
    print(f"预测标签: {pred_label}")
    print(f"知识库首行: {first_line}")
    print(f"知识库全文:\n{knowledge_text}\n{'='*50}")

    return pred_label, first_line, knowledge_text

def upload_img(image, chatbot, chat_state, img_list):
    if image is None:
        return chatbot, chat_state, img_list, gr.update(interactive=True)
    pred_label, first_line, knowledge_text = extract_knowledge_from_image(image)
    chat_state = {
        'pred_label': pred_label,
        'first_line': first_line,
        'knowledge_text': knowledge_text,
        'image': image
    }
    return [], chat_state, [], gr.update(placeholder="Please enter your question", interactive=True)

def user_ask(user_message, chatbot, chat_state):
    if not user_message.strip():
        return chatbot, chat_state, gr.update(interactive=True)
    chatbot.append([user_message, None])
    return chatbot, chat_state, gr.update(value='', interactive=True)

def bot_answer(chatbot, chat_state):
    if chat_state is None or len(chatbot) == 0:
        return chatbot, chat_state
    user_message = chatbot[-1][0]

    pred_label = chat_state['pred_label']
    first_line = chat_state['first_line']
    knowledge_text = chat_state['knowledge_text']
    image = chat_state['image']

    retrieved = retrieve_relevant_context(user_message, knowledge_text)
    context = first_line + "\n\n" + retrieved if retrieved else first_line

    print(f"===== 新一轮提问 =====")
    print(f"问题: {user_message}")
    print(f"检索到的知识:\n{context}\n{'='*50}")

    tmp_image_path = "tmp_input.jpg"
    image.save(tmp_image_path)

    answer = generate_answer(model, processor, tokenizer, tmp_image_path, user_message, context)
    chatbot[-1][1] = answer
    return chatbot, chat_state

with gr.Blocks(css="""
    .gradio-container {max-width: 1200px; margin: auto;}
    @media (max-width: 768px) {
        .gradio-container {max-width: 95%;}
    }
""") as demo:
    gr.Markdown("<h1 align='center'>🍅 Tomato Disease RAG VQA website</h1>")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Please upload images of tomato diseases")
            upload_button = gr.Button("Upload pictures and start a conversation", variant="primary")
            clear_button = gr.Button("Reset conversation")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Dialogue History")
            text_input = gr.Textbox(label="Enter your question", placeholder="Please upload the image first", interactive=False)

    chat_state = gr.State()
    img_list = gr.State([])

    upload_button.click(fn=upload_img,
                        inputs=[image_input, chatbot, chat_state, img_list],
                        outputs=[chatbot, chat_state, img_list, text_input])
    text_input.submit(fn=user_ask,
                      inputs=[text_input, chatbot, chat_state],
                      outputs=[chatbot, chat_state, text_input]).then(
                          fn=bot_answer,
                          inputs=[chatbot, chat_state],
                          outputs=[chatbot, chat_state]
                      )
    clear_button.click(lambda: ([], None, []), [], [chatbot, chat_state, img_list])
    clear_button.click(lambda: gr.update(placeholder="Please upload the image first", interactive=False), [], [text_input])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
