import os
import json
import yaml
import torch
from pytorch_lightning import Trainer
from ms2llm.lightning_module import LlamaLightningModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from ms2llm.dataset import JsonDataset
from ms2llm.peft_model import load_peft_model
from pytorch_lightning.callbacks import Callback
from pathlib import Path

torch.set_float32_matmul_precision("medium")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "0"          
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SaveLoRAEpochCallback(Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        # 当前 epoch 从 1 开始计数（PyTorch Lightning 的 epoch 从 0 开始）
        if trainer.is_global_zero:
            current_epoch = trainer.current_epoch + 1
            epoch_dir = os.path.join(self.save_dir, f"lora_epoch{current_epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            # 保存 LoRA 权重（生成 adapter_model.safetensors 和 adapter_config.json）
            pl_module.model.save_pretrained(epoch_dir)
                
            # 获取 loss 值
            train_loss = trainer.callback_metrics.get("train_loss", torch.tensor(0.0))
            val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(0.0))
            
            # 保存 loss_info.json
            loss_info = {
                "epoch": current_epoch,
                "train_loss": float(train_loss.item()),
                "val_loss": float(val_loss.item())
            }
            with open(os.path.join(epoch_dir, "loss_info.json"), "w") as f:
                json.dump(loss_info, f, indent=2)
            
            print(f"Saved LoRA weights and loss info for epoch {current_epoch} to {epoch_dir}")

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "configs" / "train_lora.yaml")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        trust_remote_code=True
    )
   
   
    # 把 chat 用到的几个特殊标记加进词表
    special_tokens_dict = {
        "additional_special_tokens": [
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|eot_id|>",
        ]
    }
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Number of added special tokens: {num_added}")
   
    model = load_peft_model(config["model_path"], tokenizer=tokenizer)

    
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id


    train_dataset = JsonDataset(
        config["train_question"],
        config["train_answer"],
        tokenizer,
        max_length=config["max_length"]
    )
    
    val_dataset = JsonDataset(
        config["val_question"],
        config["val_answer"],
        tokenizer,
        max_length=config["max_length"]
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False
    )

    lightning_model = LlamaLightningModel(
        model=model,
        tokenizer=tokenizer,
        learning_rate=config["learning_rate"]
    )


    save_lora_callback = SaveLoRAEpochCallback(
        save_dir=config["output_dir"]
    )


    trainer = Trainer(
        max_epochs=config["max_epochs"],
        precision="bf16-mixed",
        accelerator="gpu",
        devices=config["devices"],
        strategy="ddp_find_unused_parameters_false",
        callbacks=[save_lora_callback],
        default_root_dir=config["log_dir"],
        log_every_n_steps=100,
    )
        


    trainer.fit(lightning_model, train_loader, val_loader)

if __name__ == "__main__":
    main()
