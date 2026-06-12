import json
import os
import re
import glob
import yaml
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch.distributed as dist   
from pathlib import Path

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
project_root = Path(__file__).resolve().parent.parent
config = load_config(project_root / "configs" / "inference_lora.yaml")

# ====================== 初始化分布式 ======================
# ===== 先拿 local_rank，先选 GPU =====
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# ===== 再初始化分布式 =====
def init_dist():
    if dist.is_initialized():
        return
    dist.init_process_group(backend="nccl")

init_dist()
rank = dist.get_rank()
world_size = dist.get_world_size()

print(f"Using GPU {local_rank}/{world_size} ({torch.cuda.get_device_name(local_rank)})")

# ====================== 模型路径 ======================
base_model = config["base_model"]
lora_path = config["lora_path"]
output_dir = config["output_dir"]
os.makedirs(output_dir, exist_ok=True)

# ====================== 加载 tokenizer ======================
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = "left"   # decoder-only 左侧 padding

special_tokens_dict = {
    "additional_special_tokens": [
        "<|start_header_id|>user<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|eot_id|>",
    ]
}
tokenizer.add_special_tokens(special_tokens_dict)

# 确保有 pad_token（很多 LLaMA 系列默认没有）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"pad_token_id = {tokenizer.pad_token_id}, eos_token_id = {tokenizer.eos_token_id}")

# 把 eot_id 取出来，后面会同时作为 eos 备选之一
eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

# ====================== 加载模型 & LoRA ======================
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, lora_path)

model.to(device)
model.eval()

# ====================== Prompt 构造 ======================
def build_prompt(question: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

# ====================== 读取测试集 ======================
test_path = config["test_path"]

with open(test_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)

test_data = all_data
print(f"Test set size: {len(test_data)}")

# 用“按原始索引取模”的方式切分，记录 global_idx 方便还原顺序 
local_items = [
    (i, q) for i, q in enumerate(test_data)
    if i % world_size == rank
]
print(f"Rank {rank} processes {len(local_items)} samples")

# ====================== 推理参数 ======================
BATCH_SIZE = config["batch_size"]
MAX_NEW_TOKENS = config["max_new_tokens"]
USE_SAMPLING = config["use_sampling"]
MAX_LENGTH = config["max_length"]

TAIL_CUT_KEYWORDS  = [
    "useRal",      
    "import ",      
    "def ",       
    "# This function takes a JSON",
    "ніцип_results",
    "simplified_model:",
]

# ====================== 收尾清洗函数 ======================
def clean_tail(text: str) -> str:
    """
    对模型生成的原始文本做“收尾清理”：
    1. 遇到可疑关键词（代码、乱字段）就截断
    2. 去掉末尾可能残留的非 ASCII 垃圾字符
    """
    # 按关键词截断
    for kw in TAIL_CUT_KEYWORDS:
        pos = text.find(kw)
        if pos != -1:
            text = text[:pos]

    # 去掉结尾一串非 ASCII 字符（例如乱入的西里尔字母）
    text = re.sub(r"[^\x00-\x7F]+$", "", text)

    return text.strip()


# ====================== 推理 ======================
results = []
for start in tqdm(range(0, len(local_items), BATCH_SIZE),
                  desc=f"GPU {local_rank} inference"):
    batch = local_items[start:start + BATCH_SIZE]
    _, batch_questions = zip(*batch)   

    batch_prompts = [build_prompt(q) for q in batch_questions]
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=USE_SAMPLING,
            pad_token_id=tokenizer.pad_token_id,
            # 同时把模型原本的 eos 跟 <|eot_id|> 都作为“终止符”
            eos_token_id=[tokenizer.eos_token_id, eot_id],
        )

    # 用 seq_len 跳过整段输入（pad + prompt）
    seq_len = inputs["input_ids"].shape[1]

    for i, (global_idx, question) in enumerate(batch):
        gen_ids = outputs[i][seq_len:]   # 只保留真正生成部分

        # 不跳 special token，方便我们用字符串截断 <|eot_id|>
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

        # 先按 <|eot_id|> 截断（如果模型有生成）
        if "<|eot_id|>" in raw_text:
            raw_text = raw_text.split("<|eot_id|>")[0]

        # 使用收尾清洗函数去掉后面乱跑的代码/字段/乱码
        raw_text = clean_tail(raw_text)
        
        # 最后再把 special token 去掉，得到干净文本
        text = tokenizer.decode(
            tokenizer.encode(raw_text, add_special_tokens=False),
            skip_special_tokens=True
        ).strip()

        results.append({
            "index": global_idx,  
            "input": question,
            "prediction": text,
        })

# ====================== 保存 rank 文件 ======================
rank_path = os.path.join(output_dir, f"rank{rank}.jsonl")
with open(rank_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Rank {rank} results saved to {rank_path}")

# 所有 rank 写完各自文件后再合并
dist.barrier()

if rank == 0:
    print("Merging outputs from all ranks...")
    all_results = []
    for path in glob.glob(os.path.join(output_dir, "rank*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                all_results.append(json.loads(line))

    all_results.sort(key=lambda x: x["index"])
    merged_path = os.path.join(output_dir, "final_all.jsonl")
    with open(merged_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Merged output saved to {merged_path}, total samples: {len(all_results)}")

dist.destroy_process_group()
