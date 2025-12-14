import os
import torch
import csv
from datetime import datetime
from transformers import AutoProcessor, AutoTokenizer
from peft import PeftModel, PeftConfig
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM as LlavaForConditionalGeneration
import sys
sys.path.append('./clipmain')
from clip import CLIP

MERGED_MODEL_PATH = "TRAG-DPO/llava-v1.5-7b-all-dpo"
SBERT_MODEL_PATH = "TRAG-DPO/paraphrase-MiniLM-L6-v2"
KNOWLEDGE_BASE_DIR = "TRAG-DPO/know"
INPUT_CSV_PATH = "TRAG-DPO/dpo_test_notnull.csv"
OUTPUT_CSV_PATH = "output_label_answer_llava_all_dpo.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(SBERT_MODEL_PATH, device=device)

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
        "max_new_tokens": 2048,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("ASSISTANT:")[-1].strip()
import time
from tqdm import tqdm
def get_code( data_loader, length: int):
    encoder_time = 0
    valP1 = []
    valP2 = []
    
    for i, (index, image, text, label) in enumerate(tqdm(data_loader)):
        start_encoder_time = time.time()

        image = image.to(0, non_blocking=True)
        text = text.to(0, non_blocking=True)

        image_hash = torch.sign(image) 
        text_hash = torch.sign(text)
        encoder_time = time.time() - start_encoder_time
        
        valP1.append(image_hash.cpu())
        valP2.append(text_hash.cpu()) 

    img_buffer = torch.cat(valP1, dim=0)
    text_buffer = torch.cat(valP2, dim=0) 
    
    return img_buffer, text_buffer, encoder_time

def main():
    start_time = time.time()
    import numpy as np
    from torch.utils.data import DataLoader
    from dataloader import dataloader,my_dataset
    features_dict = np.load('TRAG-DPO/LLaVA-1.1.3/clipmain/features_datasets.npy', allow_pickle=True).item()

    ida_features_np = features_dict['ida_features']
    tda_features_np = features_dict['tda_features']
    lda_features_np = features_dict['lda_features']

    retrieval_labels = torch.Tensor(lda_features_np)
    retrieval_num = len(retrieval_labels)
    retrieval_loader = DataLoader(my_dataset(ida_features_np,tda_features_np, lda_features_np),
                                batch_size=512,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)
    retrieval_img, retrieval_txt, r_encoder_time = get_code(data_loader=retrieval_loader, length=retrieval_num)

    model, processor, tokenizer = load_model()
    modelclip = CLIP()

    with open(INPUT_CSV_PATH, newline='', encoding='utf-8') as infile, open(OUTPUT_CSV_PATH, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header + ['pred', 'model_answer'])

        for row in reader:
            single_start_time = time.time()
            image_path, label, question, reference = row[:4]

            import torchvision.transforms as transforms

            image_pil = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            image_tensor = transform(image_pil).unsqueeze(0).to(device).to(torch.float32)
            images_feature = modelclip.detect_image_for_eval(image_tensor, texts=None)
            ida_features_np = images_feature[0].cpu().numpy()
            query_img = torch.sign(torch.tensor(ida_features_np))
            query_labels = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).float()

            from utils.calc_utils import calc_top1_labels
            mAPi2t = calc_top1_labels(query_img, retrieval_txt, query_labels, retrieval_labels)
            y_pred = np.array([label.cpu().numpy().squeeze() for label in mAPi2t]) 
            y_pred = np.argmax(y_pred, axis=1)
            try:
                for pred in y_pred:
                    knowledge_path = get_knowledge_path(pred)
                    if not os.path.exists(knowledge_path):
                        print(f"[!] 未找到知识库文件: {knowledge_path}, 使用空上下文")
                        knowledge_text = ""
                    else:
                        with open(knowledge_path, "r", encoding="utf-8") as f:
                            first_line = f.readline().strip()
                            knowledge_text = f.read()
                    retrieved = retrieve_relevant_context(question, knowledge_text)
                    context = first_line + "\n\n" + retrieved if retrieved else first_line
                    model_answer = generate_answer(model, processor, tokenizer, image_path, question, context)

                    writer.writerow(row +[pred]+[model_answer])
                    single_end_time = time.time()
                    elapsed_single = single_end_time - single_start_time
                    print(f"时间：{elapsed_single:.2f}---✓ 问题处理完成：{question}")
            except Exception as e:
                print(f"[X] 错误: {e} (问题: {question})")
                writer.writerow(row + [f"[ERROR] {e}"])

    print(f"\n✅ 所有数据处理完成，结果保存在: {OUTPUT_CSV_PATH}")
    end_time = time.time()  # 记录结束时间
    elapsed = end_time - start_time
    print(f"总运行时间: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")

if __name__ == "__main__":
    main()
