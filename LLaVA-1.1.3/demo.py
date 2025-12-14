import os
import torch
from transformers import AutoProcessor, AutoTokenizer
from peft import PeftModel, PeftConfig
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM as LlavaForConditionalGeneration
import sys
sys.path.append('./clipmain')
from clip import CLIP
import gradio as gr
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from utils.calc_utils import calc_top1_labels

# ========= 配置 =========
MERGED_MODEL_PATH = "TRAG-DPO/llava-v1.5-7b-merged-test3"
SBERT_MODEL_PATH = "TRAG-DPO/paraphrase-MiniLM-L6-v2"
KNOWLEDGE_BASE_DIR = "TRAG-DPO/know"
FEATURES_PATH = "TRAG-DPO/LLaVA-1.1.3/clipmain/features_datasets.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(SBERT_MODEL_PATH, device=device)

# ========= 工具函数 =========
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

# ========= 加载检索特征 =========
features_dict = np.load(FEATURES_PATH, allow_pickle=True).item()
ida_features_np = features_dict['ida_features']
tda_features_np = features_dict['tda_features']
lda_features_np = features_dict['lda_features']

retrieval_labels = torch.tensor(lda_features_np, dtype=torch.float32)
retrieval_txt = torch.sign(torch.tensor(tda_features_np, dtype=torch.float32))

# ========= 主处理函数 =========
model, processor, tokenizer = load_model()
modelclip = CLIP()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def rag_pipeline(image, question):
    try:
        # 提取CLIP特征
        image_tensor = transform(image).unsqueeze(0).to(device).to(torch.float32)
        images_feature = modelclip.detect_image_for_eval(image_tensor, texts=None)
        ida_features_np = images_feature[0].cpu().numpy()
        query_img = torch.sign(torch.tensor(ida_features_np, dtype=torch.float32))
        query_labels = torch.tensor([[1] + [0] * 9], dtype=torch.float32)

        # 检索Top1标签
        mAPi2t = calc_top1_labels(query_img, retrieval_txt, query_labels, retrieval_labels)
        y_pred = np.array([label.cpu().numpy().squeeze() for label in mAPi2t])
        pred_label = int(np.argmax(y_pred, axis=1)[0])

        # 加载知识
        knowledge_path = get_knowledge_path(pred_label)
        if not os.path.exists(knowledge_path):
            knowledge_text = ""
            first_line = "[无知识]"
        else:
            with open(knowledge_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                knowledge_text = f.read()

        retrieved = retrieve_relevant_context(question, knowledge_text)
        context = first_line + "\n\n" + retrieved if retrieved else first_line

        # LLaVA回答
        image_path_tmp = "tmp.jpg"
        image.save(image_path_tmp)
        model_answer = generate_answer(model, processor, tokenizer, image_path_tmp, question, context)

        return f"预测标签: {pred_label}", f"检索到的知识:\n{context}", model_answer

    except Exception as e:
        return "[ERROR]", "", f"处理时出错: {str(e)}"

# ========= Gradio 界面 =========
with gr.Blocks() as demo:
    gr.Markdown("# 🍅 Tomato Disease RAG QA Demo")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传番茄病害图像")
            question_input = gr.Textbox(label="输入你的问题", placeholder="例如: What are the symptoms?")
            submit_btn = gr.Button("提交")
        with gr.Column():
            label_output = gr.Textbox(label="预测标签")
            context_output = gr.Textbox(label="检索到的知识", lines=10)
            answer_output = gr.Textbox(label="模型回答", lines=5)

    submit_btn.click(fn=rag_pipeline, inputs=[image_input, question_input],
                     outputs=[label_output, context_output, answer_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
