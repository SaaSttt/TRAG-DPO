import torch
from transformers import AutoTokenizer
from peft import PeftModel
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM as LlavaForConditionalGeneration

# 路径配置
BASE_MODEL_PATH = "TRAG-DPO/llava-v1.5-7b"
LORA_PATH = "TRAG-DPO/DPO-RAG-main/train/dpo/checkpoints/llava-v1.5-7b-task-lora_dpo"
MERGED_MODEL_PATH = "TRAG-DPO/llava-v1.5-7b-dpo"

def merge_lora_weights():
    """合并 LoRA 权重到基础模型，并保存"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("✅ 加载基础模型...")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    print("✅ 加载 LoRA 权重...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        is_trainable=False,
        device_map="auto",
    )

    print("✅ 合并 LoRA 到基础模型...")
    model = model.merge_and_unload()
    
    print(f"✅ 保存合并模型至：{MERGED_MODEL_PATH}")
    model.save_pretrained(MERGED_MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    print("🎉 模型合并完成，后续可直接加载合并后的模型！")

if __name__ == "__main__":
    merge_lora_weights()
