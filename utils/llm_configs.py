import torch
from trl import SFTConfig
from pydantic import BaseModel, Field
import logging
from typing import Optional, Generic, List, TypeVar, Literal
from transformers import (
    TrainingArguments,
)
import wandb

# --------------------------------------------------------------------------
# SECTION 1: CONFIGURATION (Pydantic Models)
# --------------------------------------------------------------------------

class PeftConfig(BaseModel):
    """Configuration for Parameter-Efficient Fine-Tuning (PEFT), specifically LoRA."""
    enabled: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(
        default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )
    add_eot_token: bool = False  # When True, also trains embedding and lm_head layers

class QuantizationConfig(BaseModel):
    """Configuration for model quantization. '4bit' enables QLoRA."""
    mode: Optional[Literal["4bit", "8bit"]] = None

class ModelConfig(BaseModel):
    """Top-level configuration for the model."""
    id: str = "allenai/OLMo-2-1124-7B"
    torch_dtype: str = "auto"
    attn_implementation: Optional[Literal["flash_attention_2"]] = "flash_attention_2"
    peft: PeftConfig = Field(default_factory=PeftConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    
_T_co = TypeVar("_T_co", covariant=True)

class TrainingConfig(BaseModel):
    """Configuration for the training process, aligned with HF TrainingArguments."""
    context_length: int = 4096
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16 # This gives us a effective batch size of 32
    optim: str = "paged_adamw_8bit" # Saves VRAM by using 8bit Adam
    # evaluation_strategy: str = "epoch"
    weight_decay: float = 0.1
    # max_grad_norm: float = 0.3 # defaults to 1
    gradient_checkpointing: bool = False # Saves VRAM by using gradient checkpointing
    use_liger_kernel: bool = True # This saves VRAM
        
    # These Hyperparameters are overwritten for LIMA
    num_train_epochs: int = 1
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.03
    sequential_sampling: bool = False # Random sampling is default behavior

    # Logging + Misc.
    report_to: str = "wandb"
    run_name: str = "fine_tuning"
    logging_steps: int = 1    
    logging_strategy: str = "steps"
    save_strategy: str = "no" # We'll save manually
    remove_unused_columns: bool = False
    seed: int = 42  # For reproducible results

    # SFT Config 
    completion_only_loss: Optional[bool] = None
    dataset_text_field: str = "text"
    packing: bool = True
    padding_free: bool = True # This saves VRAM (Requires Flash Attention 2)
    reverse_ffd_packing: bool = False

    def to_training_args(self) -> TrainingArguments:
        """Creates a transformers.TrainingArguments object from the config."""
        return TrainingArguments(
            max_length = self.context_length,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            optim=self.optim,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio, 
            warmup_steps=self.warmup_steps, # Handle warmup_steps if ratio is not desired (LIMA case)
            # evaluation_strategy = self.evaluation_strategy,
            # max_grad_norm=self.max_grad_norm,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) and torch.cuda.is_available(),
            use_liger_kernel=self.use_liger_kernel,
            gradient_checkpointing=self.gradient_checkpointing,
            seed=self.seed,
            remove_unused_columns=self.remove_unused_columns,
            # sequential_sampling = self.sequential_sampling,
            
            # Logging
            run_name=self.run_name,
            logging_strategy=self.logging_strategy,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            report_to=self.report_to,
        )
    def to_sft_training_args(self, sequential_sampling = False) -> TrainingArguments:
        """Creates a transformers.TrainingArguments object from the config."""
        return SFTConfig(
            dataset_text_field="text",
            packing = self.packing,
            padding_free = self.padding_free, # This saves VRAM (Requires Flash Attention 2)
            max_length = self.context_length,
            completion_only_loss = self.completion_only_loss,

            # Training Arguments
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            optim=self.optim,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            # Handle warmup_steps if ratio is not desired (LIMA case)
            warmup_steps=self.warmup_steps,
            

            # evaluation_strategy=self.evaluation_strategy,
            # max_grad_norm=self.max_grad_norm,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) and torch.cuda.is_available(),
            gradient_checkpointing=self.gradient_checkpointing,
            use_liger_kernel=self.use_liger_kernel,
            seed=self.seed,
            remove_unused_columns=self.remove_unused_columns,
            # sequential_sampling = self.sequential_sampling,
            # reverse_ffd_packing = self.reverse_ffd_packing,

            # Logging
            run_name=self.run_name,
            logging_strategy=self.logging_strategy,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            report_to=self.report_to,
        )

class InferenceConfig(BaseModel):
    """Configuration for the inference process."""
    max_new_tokens: int = 1024
    temperature: float = 0.1
    do_sample: bool = False
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 0