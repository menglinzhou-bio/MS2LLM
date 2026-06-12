import pytorch_lightning as pl
import torch
import os
from safetensors.torch import load_file  # 用于加载 .safetensors 文件
from peft import PeftModel


class LlamaLightningModel(pl.LightningModule):
    def __init__(self, model, tokenizer, learning_rate=3e-4):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['model', 'tokenizer'])
        self._optimizer = None
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )
        self._optimizer = optimizer  # 保存引用
        return optimizer
    
    def load_accelerate_checkpoint(self, checkpoint_dir):
        # === 1. 加载 LoRA 适配器权重 ===
        config_path = os.path.join(checkpoint_dir, "adapter_config.json")
        adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        
        if os.path.exists(config_path) and os.path.exists(adapter_path):
            self.model.load_adapter(checkpoint_dir, adapter_name="default")
            print("✅ LoRA loaded via load_adapter.")
        elif os.path.exists(adapter_path):
            adapter_weights = load_file(adapter_path, device="cpu")
            self.model.load_state_dict(adapter_weights, strict=False)
            print("✅ LoRA loaded via load_state_dict (fallback).")
        else:
            raise FileNotFoundError(f"LoRA weights not found in {checkpoint_dir}")



