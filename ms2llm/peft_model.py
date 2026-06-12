import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def load_peft_model(model_path, tokenizer=None):
    # 加载模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,             
        trust_remote_code=True,
    )
    
    
    # LoRA 配置
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj","v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 应用 LoRA
    model = get_peft_model(base_model, peft_config)
    
    
    model.enable_input_require_grads()  # 启用输入梯度
    
    if tokenizer is not None:
        print(f"Resizing token embeddings to {len(tokenizer)}...")
        model.resize_token_embeddings(len(tokenizer))
        

    return model
