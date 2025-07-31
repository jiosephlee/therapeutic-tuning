import sys 
#sys.path.append('../../trl')

import math
import os
import wandb
import torch
from typing import Optional, List, Literal
import pandas as pd
import matplotlib.pyplot as plt

# Third-party imports
from datasets import Dataset, load_dataset
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from liger_kernel.transformers import LigerCrossEntropyLoss
from trl import SFTConfig, SFTTrainer
from utils.llm_configs import PeftConfig, ModelConfig, TrainingConfig, InferenceConfig
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, Any

# --------------------------------------------------------------------------
# SECTION 2: CORE LLM OPERATIONS
# --------------------------------------------------------------------------

def create_peft_model_for_training(model, log, config: PeftConfig):

    model = prepare_model_for_kbit_training(model)
    log.info("Applying PEFT (LoRA)...")
    # Prepare modules_to_save for instruction tuning
    modules_to_save = ["lm_head", "embed_tokens"] if config.add_eot_token else None
                
    peft_config = PeftLoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,  # Add this line
    )
    model = get_peft_model(model, peft_config)
    log.info("LoRA applied. Trainable parameters:")
    model.print_trainable_parameters()

    log.info("PEFT Model Created successfully.")
    return model
    
def load_model_for_training(config: ModelConfig, log, use_cpu_and_gpu = False, add_special_token = None, use_existing_lima_tokenizer=False, use_existing_lima_model = False):
    """
    Loads a model and tokenizer for training, applying quantization and PEFT.
    **ENHANCED** with robust QLoRA setup from open-instruct.
    """
    log.info(f"Loading model '{config.id}' for training...")

    # Determine torch dtype
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    quant_config = None
    if config.quantization.mode == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, # Use bfloat16 for compute
            bnb_4bit_use_double_quant=True,
        )
    elif config.quantization.mode == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        if quant_config is None:
            model = AutoModelForCausalLM.from_pretrained(
                config.id,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map='auto' if use_cpu_and_gpu else "cuda",
                attn_implementation=config.attn_implementation,
            )
        else:
            print("...Quantizing...")
            model = AutoModelForCausalLM.from_pretrained(
            config.id,
            trust_remote_code=True,
            torch_dtype=dtype,
            quantization_config=quant_config,
            device_map='auto' if use_cpu_and_gpu else "cuda", #Assume we're operating in a low VRAM environment since we're quantizing
            attn_implementation=config.attn_implementation,
        )
        if use_existing_lima_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained("jiosephlee/olmo2-lima", trust_remote_code=True)
            model.resize_token_embeddings(len(tokenizer))
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.id, trust_remote_code=True, use_fast=True)
            # Add special tokens before doing PEFT
            if add_special_token is not None:
                log.info(f"Adding special token: {add_special_token}")
                special_tokens_dict = {'additional_special_tokens': [add_special_token]}
                tokenizer.add_special_tokens(special_tokens_dict)  
                model.resize_token_embeddings(len(tokenizer))
            
        # Crucial step for preparing a quantized model for PEFT training.
        if config.quantization.mode:
            model = prepare_model_for_kbit_training(model)

        if config.peft.enabled:
            if use_existing_lima_model:
                model.load_adapter("jiosephlee/olmo2-lima", adapter_name="lima")
            else:
                log.info("Applying PEFT (LoRA)...")
                # Prepare modules_to_save for instruction tuning
                modules_to_save = ["lm_head", "embed_tokens"] if config.peft.add_eot_token else None
                
                peft_config = PeftLoraConfig(
                    r=config.peft.lora_r,
                    lora_alpha=config.peft.lora_alpha,
                    lora_dropout=config.peft.lora_dropout,
                    target_modules=config.peft.target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                    modules_to_save=modules_to_save,  # Add this line
                )
                model = get_peft_model(model, peft_config)
                log.info("LoRA applied. Trainable parameters:")
                model.print_trainable_parameters()

    log.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def prepare_lima_dataset(tokenizer: AutoTokenizer, log, use_eot_token=False, sort=False):
    """
    Loads the GAIR/lima dataset, and formats
    the conversations into a text format suitable for SFTTrainer.

    Args:
        tokenizer: The tokenizer to modify.
        model: The model to resize embeddings for.

    Returns:
        A tuple of (train_dataset, eval_dataset).
    """
    log.info("Preparing GAIR/lima dataset...")
    EOT_TOKEN = "<|EOT|>"
    if not use_eot_token:
        EOT_TOKEN = "\nResponse:"
    # 2. Load the dataset
    dataset = load_dataset("GAIR/lima")
    # The paper uses 1000 for training, 50 for dev. The HF dataset has 1030 train examples.
    # We'll split it accordingly.
    train_dataset = dataset["train"].shuffle(seed=42)
    # train_dataset = full_train_dataset
    log.info(f"{len(train_dataset)} training examples.")

    # 3. Define the formatting function
    def format_lima_conversation(example):
        conversation = example['conversations']
        # Join turns with the EOT token. Add one at the very end.
        formatted_text = f"{EOT_TOKEN}".join(conversation) + tokenizer.eos_token
        return {"text": formatted_text}

        
    # 4. Apply the formatting
    train_dataset = train_dataset.map(format_lima_conversation, remove_columns=['conversations', 'source'])

    if sort:
        # add a temporary 'length' column
        train_dataset = train_dataset.map(
            lambda x: {"_len": len(x["text"])},
            desc="Computing lengths for sort",
        )
        # Dataset.sort always sorts ascending, so we reverse afterwards
        train_dataset = (
            train_dataset
            .sort("_len")                                      # shortest → longest
            .select(list(range(len(train_dataset) - 1, -1, -1)))  # flip order
            .remove_columns("_len")                            # clean-up
        )
        log.info("Training set sorted by descending length.")
        
    return train_dataset

# **IMPORTANT** Custom trainer to use 'sum' loss, a best practice for chat models.
# No longer does this but we leave it here in case we want to go back to it. This is functionally just a SFTTrainer
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, use_liger_loss, *args, **kwargs):
        super().__init__(*args, **kwargs) # Pass all remaining args/kwargs to parent
        self.use_liger_loss = use_liger_loss
    
def fine_tune_on_text(
    model, tokenizer, log, text_content: str, train_cfg: TrainingConfig, *, train=True, tag: str = "finetuning on text...", callbacks: Optional[List[TrainerCallback]] = None
):
    """
    Fine-tunes a model on a given string of text by chunking it properly.
    
    Args:
        model: The model to fine-tune
        tokenizer: The tokenizer
        text_content: The text to fine-tune on
        train_cfg: Training configuration
        tag: Tag for logging
        callbacks: Optional list of TrainerCallbacks to add to the trainer.
    """
    log.info(f"Starting SFT for '{tag}'...")
    
    text_content = text_content + tokenizer.eos_token
    text_chunks, num_tokens = chunk_text(text_content, tokenizer, train_cfg.context_length)
    
    log.info(f"[{tag}] Tokens: {num_tokens}, Context: {train_cfg.context_length} -> {len(text_chunks)} chunks")
    
    dataset = Dataset.from_dict({"text": text_chunks})
    log.info(f"[{tag}] Created dataset with {len(text_chunks)} chunks (including the eos token)")
    
    assert(train_cfg.gradient_accumulation_steps * train_cfg.per_device_train_batch_size == len(text_chunks))
    log.info(f"[{tag}] Gradient_accumulation_steps ({train_cfg.gradient_accumulation_steps}) * {train_cfg.per_device_train_batch_size} = {len(text_chunks)}")
    
    training_args = train_cfg.to_sft_training_args() 
    
    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        use_liger_loss=True,
        callbacks=callbacks
    )
    
    if train:
        trainer.train()
        log.info(f"SFT complete for '{tag}'.")
        wandb.finish()
    return trainer

def fine_tune_on_texts(
    model, tokenizer, log, texts: List[str], train_cfg: TrainingConfig, *, train=True, tag: str = "finetuning on texts...", callbacks: Optional[List[TrainerCallback]] = None
):
    """
    Fine-tunes a model on a given list of texts by chunking them and training on all chunks together.
    
    Args:
        model: The model to fine-tune.
        tokenizer: The tokenizer.
        texts: The list of text strings to fine-tune on.
        train_cfg: Training configuration.
        tag: Tag for logging.
        callbacks: Optional list of TrainerCallbacks to add to the trainer.
    """
    log.info(f"Starting SFT for '{tag}' on {len(texts)} documents...")

    # Add EOS token to each text
    for i in range(len(texts)):
        texts[i] = texts[i] + tokenizer.eos_token 
    
    # Chunk the texts into smaller pieces based on context length
    all_text_chunks, total_tokens = chunk_texts(texts, tokenizer, train_cfg.context_length)

    log.info(f"[{tag}] Total tokens: {total_tokens}, Context: {train_cfg.context_length} -> {len(all_text_chunks)} total chunks")
    
    # Create dataset with all chunked texts
    dataset = Dataset.from_dict({"text": all_text_chunks})
    
    assert(train_cfg.gradient_accumulation_steps <= len(all_text_chunks))
    log.info(f"[{tag}] Gradient_accumulation_steps ({train_cfg.gradient_accumulation_steps}) is less than {len(all_text_chunks)}")

    training_args = train_cfg.to_sft_training_args() # Packing is False to avoid document re-ordering and padding free is false to avoid OOM issues

    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        use_liger_loss=True,
        callbacks=callbacks
    )
    if train:
        trainer.train()
        log.info(f"SFT complete for '{tag}'.")
        wandb.finish()
    return trainer

def sft_train_on_dataset(
    model,  tokenizer, log, train_dataset: Dataset, train_cfg: TrainingConfig, train=True, use_liger_loss =False, callbacks: Optional[List[TrainerCallback]] = None
):
    """
    A generalized function to run SFT on a prepared dataset. Effective batch size is batch_size (2) * gradient_accumulation_steps (16) = 32 as per LIMA
    """
    log.info("Starting SFT training run...")
    training_args = train_cfg.to_sft_training_args()

    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        processing_class=tokenizer,
        use_liger_loss = use_liger_loss,
        callbacks=callbacks
    )

    if train:
        trainer.train()
        log.info("SFT training complete.")
        wandb.finish()
    return trainer

def save_model(model, tokenizer, log, save_path: str):
    """
    Saves the model and tokenizer. 
    """ # If LoRA was used, it merges the adapters into the base model for easy deployment.
    os.makedirs(save_path, exist_ok=True)
    # if hasattr(model, "merge_and_unload"):
    #     log.info("Merging LoRA adapters and saving full model...")
    #     model = model.merge_and_unload()
    # else:
    log.info("Saving full model...")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    log.info(f"Model saved to {save_path}")
    

class MedexKnowledgeProbeCallback(TrainerCallback):
    """
    A callback designed for the MEDEX dataset to evaluate perplexity and log probability
    on specific knowledge probes. It tracks metrics and their deltas from a pre-training baseline.

    It performs two forward passes to measure performance with different contexts:
    1. Full context (Entity, SMILES, Fact)
    2. Minimal context (Entity, Fact)
    """
    def __init__(self, tokenizer: AutoTokenizer, probe_dataset_path: str, batch_size: int = 8, log_prefix="medex_probe"):
        self.tokenizer = tokenizer
        self.log_prefix = log_prefix
        self.batch_size = batch_size

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        df = pd.read_csv(probe_dataset_path)

        # Prepare all probe text variations and contexts.
        # The `.get(c, pd.Series(index=df.index, dtype=object).fillna(""))` provides a fallback for missing columns.
        self.probes = {
            "text": df["text"].tolist(),
            "fact": df["fact"].tolist(),
            "fact_target": df["fact_target"].tolist(),
            "full_context_probe": df["full_context_probe"].tolist(),
            "minimal_context_probe": df["minimal_context_probe"].tolist(),
            "full_context": df.get("full_context").tolist(),
            "minimal_text": df.get("minimal_text").tolist(),
            "minimal_context": df.get("minimal_context").tolist(),
            "empty_context": [""] * len(df)
        }
        self.probe_indices = df.index.tolist()

        # Dynamically determine max_length from the longest probe text
        all_probe_texts = self.probes['text'] 
        max_len = 0
        for text in all_probe_texts:
            token_len = len(self.tokenizer(str(text), add_special_tokens=False)['input_ids'])
            if token_len > max_len:
                max_len = token_len

        # Set max_length to the calculated value, padding to a multiple of 8 for performance.
        self.max_length = (max_len + 7) // 8 * 8
        print(f"INFO: MedexKnowledgeProbeCallback dynamically set max_length to {self.max_length}")


        # Calculate entity frequencies and assign buckets for analysis
        entity_counts = df['entity'].value_counts()
        self.entity_freq_map = df['entity'].map(entity_counts)
        bins = [0, 5, 10, 20, 30, 40, 50, 100, float('inf')]
        labels = ['1-5', '6-10', '11-20', '21-30', '31-40', '41-50', '51-100', '100+']
        self.entity_freq_buckets = pd.cut(self.entity_freq_map, bins=bins, labels=labels, right=True)

        self.history = []
        self.baseline_metrics = {}

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        print("MedexKnowledgeProbeCallback: Calculating baseline metrics before training...")
        self.baseline_metrics = self._evaluate_probes(model)
        print("MedexKnowledgeProbeCallback: Baseline metrics calculation complete.")
        # Log baseline metrics to wandb if desired
        log_data = {f"{self.log_prefix}/{k}": v for k, v in self.baseline_metrics.items() if "avg" in k}
        if state.is_world_process_zero:
            wandb.log(log_data, step=0)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        step_metrics = self._evaluate_probes(model)

        # Calculate deltas and prepare for logging
        log_data = {}
        results_with_deltas = {}
        # Use the keys from the step_metrics to ensure alignment
        for key in step_metrics.keys():
             if not isinstance(step_metrics[key], list): # Skip non-list items like '..._avg_...'
                 continue
             
             step_value_list = step_metrics[key]
             baseline_value_list = self.baseline_metrics.get(key, [0] * len(step_value_list))
             
             # Store the raw metric
             results_with_deltas[key] = step_value_list
             
             # Calculate and store the delta
             delta_list = [s - b for s, b in zip(step_value_list, baseline_value_list)]
             delta_key = f"{key}_delta"
             results_with_deltas[delta_key] = delta_list

        step_results = {'step': state.global_step, **results_with_deltas}
        self.history.append(step_results)

        # Log average metrics to wandb
        avg_log_data = {}
        for key, value in step_metrics.items():
            if "avg" in key:
                baseline_value = self.baseline_metrics.get(key, 0)
                delta = value - baseline_value
                avg_log_data[f"{self.log_prefix}/{key}"] = value
                avg_log_data[f"{self.log_prefix}/{key}_delta"] = delta

        if state.is_world_process_zero:
            wandb.log(avg_log_data, step=state.global_step)

    def _get_target_mask(self, tokenized_full, context_lengths, target_lengths, full_lengths):
        """Identifies the token positions of the target sequence within the full sequence."""
        mask = torch.zeros_like(tokenized_full, dtype=torch.bool)

        for i in range(tokenized_full.shape[0]):
            # Assert that context_length + target_length = full_length for each item
            expected_full_length = context_lengths[i].item() + target_lengths[i].item()
            actual_full_length = full_lengths[i].item() + 1
            assert expected_full_length == actual_full_length, \
                f"Length mismatch at index {i}: context ({context_lengths[i].item()}) + target ({target_lengths[i].item()}) = {expected_full_length} != full ({actual_full_length})"
            
            start_idx = int(context_lengths[i].item())
            
            # Calculate the end index and cap it at the actual sequence length for safety.
            end_idx = int(min(start_idx + target_lengths[i].item(), full_lengths[i].item())) # TODO: Hm is it right to cap by full_lengths? in index 0 case, we're maxing it at 41
            # print(start_idx)
            # print(end_idx)
            if start_idx < end_idx:
                mask[i, start_idx:end_idx] = True
            
        return mask

    def _run_pass_and_get_metrics(self, model, full_text_key: str, probes_config: Dict[str, Dict[str, str]]):
        device = model.device
        full_texts_list = self.probes[full_text_key]
        
        results_aggregator = {
            probe_name: {'losses': [], 'counts': [], 'hits_at_1': [], 'hits_at_5': [], 'hits_at_10': []}
            for probe_name in probes_config.keys()
        }

        for i in range(0, len(full_texts_list), self.batch_size):
            batch_texts = [str(t) for t in full_texts_list[i:i + self.batch_size]]
            if not batch_texts: continue

            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()

            loss_fct = LigerCrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_per_token = loss_per_token.view(input_ids.size(0), -1)

            for probe_name, config in probes_config.items():
                context_key = config["context_key"]
                target_key = config["target_key"]

                batch_contexts = [str(c) for c in self.probes[context_key][i:i + self.batch_size]]
                batch_targets = [str(t) for t in self.probes[target_key][i:i + self.batch_size]]

                tokenized_contexts = self.tokenizer(batch_contexts, return_tensors="pt", padding=True, add_special_tokens=False)
                tokenized_targets = self.tokenizer(batch_targets, return_tensors="pt", padding=True, add_special_tokens=False)

                context_lengths = tokenized_contexts['attention_mask'].sum(dim=1)
                target_lengths = tokenized_targets['attention_mask'].sum(dim=1)
                full_lengths = shift_attention_mask.sum(dim=1)
                # print("Probe name: " + probe_name)
                # print("Texts: ")
                # print(batch_texts)
                # print("Contexts: ")
                # print(batch_contexts)
                # print("Targets: ")
                # print(batch_targets)

                target_mask = self._get_target_mask(shift_labels, context_lengths, target_lengths, full_lengths)

                assert not (target_mask.to(device) & (shift_attention_mask == 0)).any(), \
                    f"Probe '{probe_name}': Target mask and attention mask conflict."

                final_mask = shift_attention_mask * target_mask.to(device)
                #print(shift_attention_mask)
                #print(loss_per_token)
                #print(shift_attention_mask.shape)
                #print(loss_per_token.shape)
                masked_loss = loss_per_token * final_mask
                
                sum_loss = masked_loss.sum(dim=1)
                num_tokens = final_mask.sum(dim=1)
                
                results_aggregator[probe_name]['losses'].append(sum_loss)
                results_aggregator[probe_name]['counts'].append(num_tokens)

                # --- Hits @ k calculation (only for specific target probes) ---
                if config["target_key"] == "fact_target":
                    batch_hits_at_1, batch_hits_at_5, batch_hits_at_10 = [], [], []
                    for j in range(shift_labels.shape[0]): # Iterate over items in the batch
                        if target_lengths[j].item() > 0:
                            start_pos = context_lengths[j].item()
                            
                            # Ensure start_pos is a valid index within the possibly truncated sequence
                            if start_pos < full_lengths[j]:
                                actual_token_id = shift_labels[j, start_pos]
                                logits_slice = shift_logits[j, start_pos, :]
                                top_10_indices = torch.topk(logits_slice, 10).indices

                                batch_hits_at_1.append(1 if actual_token_id == top_10_indices[0] else 0)
                                batch_hits_at_5.append(1 if actual_token_id in top_10_indices[:5] else 0)
                                batch_hits_at_10.append(1 if actual_token_id in top_10_indices else 0)
                            else:
                                # This case can occur if truncation removes the target's first token.
                                batch_hits_at_1.append(0)
                                batch_hits_at_5.append(0)
                                batch_hits_at_10.append(0)
                        else:
                            # No target tokens, so no hit is possible.
                            batch_hits_at_1.append(0)
                            batch_hits_at_5.append(0)
                            batch_hits_at_10.append(0)
                
                    results_aggregator[probe_name]['hits_at_1'].append(torch.tensor(batch_hits_at_1, device=device))
                    results_aggregator[probe_name]['hits_at_5'].append(torch.tensor(batch_hits_at_5, device=device))
                    results_aggregator[probe_name]['hits_at_10'].append(torch.tensor(batch_hits_at_10, device=device))
        
        return self._aggregate_and_calculate_metrics(results_aggregator)

    def _aggregate_and_calculate_metrics(self, results_aggregator: Dict) -> Dict:
        """Aggregates batch results and computes final metrics for perplexity, log_prob, and hits@k."""
        final_metrics = {}
        for probe_name, data in results_aggregator.items():
            # --- Perplexity and Log Probability (calculated for all probes) ---
            total_loss = torch.cat(data['losses'])
            total_tokens = torch.cat(data['counts'])
            
            probe_metrics = self._calculate_perplexity_metrics(total_loss, total_tokens)
            for metric_name, value in probe_metrics.items():
                final_metrics[f"{probe_name}_{metric_name}"] = value

            # --- Hits @ k (calculated only if data was collected) ---
            if data['hits_at_1']:
                for k in [1, 5, 10]:
                    hits_key = f'hits_at_{k}'
                    all_hits = torch.cat(data[hits_key])
                    final_metrics[f"{probe_name}_{hits_key}"] = all_hits.cpu().tolist()
                    
                    avg_hits = all_hits.float().mean().item()
                    final_metrics[f"{probe_name}_avg_{hits_key}"] = avg_hits
            
        return final_metrics

    def _calculate_perplexity_metrics(self, total_loss: torch.Tensor, total_tokens: torch.Tensor) -> Dict[str, Any]:
        """Calculates perplexity, log probability, and their averages from loss and token counts."""
        metrics = {}
        
        # Note: Direct division can result in NaN/Inf if total_tokens contains zeros.
        # This is intentional to make data or tokenization issues apparent.
        mean_loss = total_loss / total_tokens

        perplexity = torch.exp(mean_loss)
        
        metrics["perplexity"] = perplexity.cpu().tolist()
        metrics["log_prob"] = mean_loss.cpu().tolist()
        
        # For averages, we still filter out NaN/Inf to get a meaningful summary of valid probes.
        valid_perplexity = perplexity[~torch.isinf(perplexity) & ~torch.isnan(perplexity)]
        valid_log_prob = mean_loss[~torch.isinf(mean_loss) & ~torch.isnan(mean_loss)]
        
        metrics["avg_perplexity"] = valid_perplexity.mean().item() if len(valid_perplexity) > 0 else 0.0
        metrics["avg_log_prob"] = valid_log_prob.mean().item() if len(valid_log_prob) > 0 else 0.0
        
        return metrics

    def _evaluate_probes(self, model):
        was_training = model.training
        model.eval()
        all_metrics = {}

        # --- Pass 1: Full Context ---
        full_context_probes_config = {
            "text": {"context_key": "empty_context", "target_key": "text"},
            "fact_given_full_context": {"context_key": "full_context", "target_key": "fact"},
            "fact_target_given_full_context_probe": {"context_key": "full_context_probe", "target_key": "fact_target"}
        }
        metrics_pass_1 = self._run_pass_and_get_metrics(model, "text", full_context_probes_config)
        all_metrics.update(metrics_pass_1)

        # --- Pass 2: Minimal Context ---
        minimal_context_probes_config = {
            "fact_given_minimal_context": {"context_key": "minimal_context", "target_key": "fact"},
            "fact_target_given_minimal_context_probe": {"context_key": "minimal_context_probe", "target_key": "fact_target"}
        }
        metrics_pass_2 = self._run_pass_and_get_metrics(model, "minimal_text", minimal_context_probes_config)
        all_metrics.update(metrics_pass_2)

        if was_training:
            model.train()
        return all_metrics

    def get_results_dataframe(self):
        records = []
        all_metrics = set()
        
        for entry in self.history:
            step = entry['step']
            num_probes = len(self.probe_indices)
            for i in range(num_probes):
                record = {
                    'step': step,
                    'probe_index': self.probe_indices[i],
                    'entity_freq_bucket': self.entity_freq_buckets[i],
                }
                # Dynamically add all available per-probe metrics from the history entry.
                # This is robust to metrics that are not calculated for every probe type.
                for key, value in entry.items():
                    if isinstance(value, list) and i < len(value):
                        record[key] = value[i]
                        all_metrics.add(key)
                records.append(record)
        
        print(f"Tracked metrics: {sorted(all_metrics)}")
        return pd.DataFrame(records)

    def plot_average_perplexities(self, output_dir="."):
        df = self.get_results_dataframe()
        avg_df = df.groupby('step').mean(numeric_only=True)

        # Select perplexity columns to plot
        metrics_to_plot = [col for col in avg_df.columns if 'perplexity' in col and 'delta' not in col]

        plt.figure(figsize=(15, 10))
        for metric in metrics_to_plot:
            plt.plot(avg_df.index, avg_df[metric], label=metric)

        plt.title('Average Perplexity Over Training Steps')
        plt.xlabel('Training Step')
        plt.ylabel('Perplexity')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "average_perplexities.png"))
        plt.close()

    def plot_perplexity_by_entity_frequency(self, output_dir="."):
        df = self.get_results_dataframe()
        
        plot_metrics = [
            'fact_target_given_full_context_probe_perplexity',
            'fact_target_given_minimal_context_probe_perplexity'
        ]

        for probe_type in plot_metrics:
            # Need to handle potential CategoricalDtype for groupby
            if pd.api.types.is_categorical_dtype(df['entity_freq_bucket']):
                df['entity_freq_bucket'] = df['entity_freq_bucket'].astype(str)
            
            grouped_df = df.groupby(['entity_freq_bucket', 'step'])[probe_type].mean().reset_index()

            plt.figure(figsize=(14, 9))
            # Sort buckets for consistent plotting order
            sorter = ['1-5', '6-10', '11-20', '21-30', '31-40', '41-50', '51-100', '100+']
            grouped_df.entity_freq_bucket = pd.Categorical(grouped_df.entity_freq_bucket, categories=sorter, ordered=True)
            grouped_df = grouped_df.sort_values('entity_freq_bucket')

            for bucket, data in grouped_df.groupby('entity_freq_bucket'):
                plt.plot(data['step'], data[probe_type], label=f'Freq: {bucket}')

            plt.title(f'{probe_type.replace("_", " ").title()} by Entity Frequency')
            plt.xlabel('Training Step')
            plt.ylabel('Average Perplexity')
            plt.legend(title="Entity Frequency Bucket")
            plt.grid(True)
            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{probe_type}_by_entity_frequency.png"))
            plt.close()

    def plot_delta_perplexity_comparison(self, output_dir="."):
        """
        Plots a comparison of delta perplexities for various probe configurations.
        - 'text' perplexity is a solid line.
        - 'fact' perplexities are dotted lines.
        - 'fact_target' perplexities are dashed lines.
        """
        df = self.get_results_dataframe()
        avg_df = df.groupby('step').mean(numeric_only=True)

        plt.figure(figsize=(15, 10))

        metrics_to_plot = {
            'text_perplexity_delta': {'label': 'Text (Overall)', 'linestyle': '-'},
            'fact_given_full_context_perplexity_delta': {'label': 'Fact (Full Context)', 'linestyle': ':'},
            'fact_target_given_full_context_probe_perplexity_delta': {'label': 'Fact Target (Full Context)', 'linestyle': '--'},
            'fact_given_minimal_context_perplexity_delta': {'label': 'Fact (Minimal Context)', 'linestyle': ':'},
            'fact_target_given_minimal_context_probe_perplexity_delta': {'label': 'Fact Target (Minimal Context)', 'linestyle': '--'},
        }

        for metric, style in metrics_to_plot.items():
            if metric in avg_df.columns:
                plt.plot(avg_df.index, avg_df[metric], label=style['label'], linestyle=style['linestyle'])

        plt.title('Average Delta Perplexity Over Training Steps')
        plt.xlabel('Training Step')
        plt.ylabel('Delta Perplexity (vs. Baseline)')
        plt.legend(title="Probe Type")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "delta_perplexity_comparison.png"))
        plt.close()

    def plot_focused_delta_perplexity(self, output_dir="."):
        """
        Plots a focused comparison of delta perplexities for the 'fact_target' probes,
        including standard deviation as a shaded area.
        """
        df = self.get_results_dataframe()
        
        metrics_to_plot = {
            'fact_target_given_full_context_probe_perplexity_delta': {'label': 'Fact Target (Full Context)', 'linestyle': '--'},
            'fact_target_given_minimal_context_probe_perplexity_delta': {'label': 'Fact Target (Minimal Context)', 'linestyle': '--'},
        }

        cols_to_agg = list(metrics_to_plot.keys())
        stats_df = df.groupby('step')[cols_to_agg].agg(['mean', 'std'])

        plt.figure(figsize=(15, 10))

        for metric, style in metrics_to_plot.items():
            if (metric, 'mean') in stats_df.columns:
                mean = stats_df[(metric, 'mean')]
                std = stats_df[(metric, 'std')].fillna(0)
                plt.plot(stats_df.index, mean, label=style['label'], linestyle=style['linestyle'])
                plt.fill_between(stats_df.index, mean - std, mean + std, alpha=0.2)

        plt.title('Fact Target Delta Perplexity: Full vs. Minimal Context')
        plt.xlabel('Training Step')
        plt.ylabel('Delta Perplexity (vs. Baseline)')
        plt.legend(title="Probe Type")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "focused_delta_perplexity_with_std.png"))
        plt.close()

    def plot_hits_at_k_for_fact_targets(self, output_dir="."):
        """
        Plots Hits@1, Hits@5, and Hits@10 for the full-context fact target probe.
        """
        df = self.get_results_dataframe()
        avg_df = df.groupby('step').mean(numeric_only=True)

        plt.figure(figsize=(15, 10))

        metrics_to_plot = {
            'fact_target_given_full_context_probe_hits_at_1': {'label': 'Hits@1'},
            'fact_target_given_full_context_probe_hits_at_5': {'label': 'Hits@5'},
            'fact_target_given_full_context_probe_hits_at_10': {'label': 'Hits@10'},
        }

        for metric, style in metrics_to_plot.items():
            if metric in avg_df.columns:
                plt.plot(avg_df.index, avg_df[metric], label=style['label'])

        plt.title('Average Hits@k for Fact Targets (Full Context)')
        plt.xlabel('Training Step')
        plt.ylabel('Hit Rate')
        plt.legend(title="Metric")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "hits_at_k_fact_targets.png"))
        plt.close()

    def plot_final_perplexity_by_bucket(self, output_dir="."):
        """
        Plots the average final delta perplexity for fact target probes,
        grouped by entity frequency bucket, with standard deviation shadows.
        """
        df = self.get_results_dataframe()
        if df.empty or 'step' not in df.columns:
            print("INFO: Cannot generate final perplexity plot, dataframe is empty or missing 'step' column.")
            return

        last_step = df['step'].max()
        df_last_step = df[df['step'] == last_step].copy()

        plt.figure(figsize=(14, 9))
        
        metrics_to_plot = {
            'fact_target_given_full_context_probe_perplexity_delta': 'Full Context',
            'fact_target_given_minimal_context_probe_perplexity_delta': 'Minimal Context'
        }

        if pd.api.types.is_categorical_dtype(df_last_step['entity_freq_bucket']):
            df_last_step['entity_freq_bucket'] = df_last_step['entity_freq_bucket'].astype(str)

        for metric, label in metrics_to_plot.items():
            if metric not in df_last_step.columns:
                continue

            bucket_stats = df_last_step.groupby('entity_freq_bucket')[metric].agg(['mean', 'std']).reset_index()

            sorter = ['1-5', '6-10', '11-20', '21-30', '31-40', '41-50', '51-100', '100+']
            bucket_stats['entity_freq_bucket'] = pd.Categorical(bucket_stats['entity_freq_bucket'], categories=sorter, ordered=True)
            bucket_stats = bucket_stats.sort_values('entity_freq_bucket')
            bucket_stats['std'] = bucket_stats['std'].fillna(0)

            x_axis = bucket_stats['entity_freq_bucket'].astype(str)
            mean = bucket_stats['mean']
            std = bucket_stats['std']

            plt.plot(x_axis, mean, label=label)
            plt.fill_between(x_axis, mean - std, mean + std, alpha=0.2)
        
        plt.title('Final Delta Perplexity of Fact Targets by Entity Frequency')
        plt.xlabel('Entity Frequency Bucket')
        plt.ylabel('Average Delta Perplexity at Final Step')
        plt.legend(title="Context Type")
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(rotation=45)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "final_delta_perplexity_by_bucket.png"))
        plt.close()


class TrainingLossPerplexityCallback(TrainerCallback):
    """
    A callback that captures the training loss at each logging step,
    calculates perplexity from it, logs it to Weights & Biases,
    and stores it for external analysis.
    This represents the perplexity of the specific data chunk seen in that step.
    """
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # The 'loss' key is only present during training steps.
        if logs is not None and 'loss' in logs:
            if state.is_world_process_zero:
                # The 'loss' is the average cross-entropy loss for the batch.
                # Perplexity is the exponentiation of this loss.
                chunk_perplexity = math.exp(logs['loss'])
                self.history.append({'step': state.global_step, 'chunked_perplexity': chunk_perplexity})
                wandb.log({"train/chunked_perplexity": chunk_perplexity}, step=state.global_step+1)
    
    def get_results_as_dataframe(self):
        """
        Returns the collected training loss perplexity data as a pandas DataFrame.
        """
        return pd.DataFrame(self.history)


def chunk_texts(texts: List[str], tokenizer, context_length: int) -> tuple[List[str], int]:
    """
    Chunks a list of texts into smaller pieces based on context length.
    
    Args:
        texts: List of text strings to chunk
        tokenizer: The tokenizer to use
        context_length: Maximum tokens per chunk
        
    Returns:
        Tuple of (all_text_chunks, total_tokens)
    """
    all_text_chunks = []
    total_tokens = 0
    
    for text_content in texts:
        tokens = tokenizer(text_content, add_special_tokens=False, truncation=False)["input_ids"]
        num_tokens = len(tokens)
        total_tokens += num_tokens
        num_chunks = math.ceil(num_tokens / context_length)
        
        for i in range(num_chunks):
            start_idx = i * context_length
            end_idx = min((i + 1) * context_length, num_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            # print(chunk_tokens[end_idx-1-start_idx])
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=False)
            all_text_chunks.append(chunk_text)
    
    return all_text_chunks, total_tokens

def chunk_text(text_content: str, tokenizer, context_length: int) -> tuple[List[str], int]:
    """
    Chunks a single text into smaller pieces based on context length.
    
    Args:
        text_content: Text string to chunk
        tokenizer: The tokenizer to use
        context_length: Maximum tokens per chunk
        
    Returns:
        Tuple of (text_chunks, num_tokens)
    """
    tokens = tokenizer(text_content, add_special_tokens=False, truncation=False)["input_ids"]
    num_tokens = len(tokens)
    num_chunks = math.ceil(num_tokens / context_length)
    
    text_chunks = []
    for i in range(num_chunks):
        start_idx = i * context_length
        end_idx = min((i + 1) * context_length, num_tokens)
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=False)
        text_chunks.append(chunk_text)
    
    return text_chunks, num_tokens

@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, config: InferenceConfig) -> str:
    """Simple inference function using Hugging Face transformers.generate."""
    inputs = tokenizer(prompt , return_tensors="pt").to(model.device)
    if config.do_sample:
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size = config.no_repeat_ngram_size
        )
    else:
        outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        repetition_penalty=config.repetition_penalty,
        no_repeat_ngram_size = config.no_repeat_ngram_size
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

@torch.inference_mode()
def analyze_text_generation(model, tokenizer, prompt, device, max_new_tokens=1024):
    """
    Generates text from a prompt and analyzes the top 5 token choices at each step.

    Args:
        model_name (str): The name of the pretrained model to use (e.g., "gpt2").
        prompt (str): The input text to generate from.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: A formatted string detailing the generation process.
    """
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text and get scores
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
    )
    print(f"Output: {tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)}\n")
          
    # Get the generated token IDs, excluding the input prompt's tokens
    generated_token_ids = outputs.sequences[0, inputs.input_ids.shape[-1]:]
    # Get the scores for each generation step
    token_scores = outputs.scores

    # --- Formatting the Output ---
    output_string = ""
    # Iterate through each generated token and its corresponding scores
    for i, generated_token_id in enumerate(generated_token_ids):
        # Get the scores for the current step
        step_scores = token_scores[i][0]

        # Apply softmax to convert logits to probabilities
        step_probs = torch.nn.functional.softmax(step_scores, dim=0)
        
        # Get the top 5 tokens and their probabilities
        top_5_probs, top_5_indices = torch.topk(step_probs, 5)

        # Decode the generated token and the top 5 tokens
        generated_token = tokenizer.decode(generated_token_id)
        
        # Get the probability of the actual chosen token
        chosen_token_prob = step_probs[generated_token_id].item()

        output_string += f'➡️ Generated Token #{i+1}: "{generated_token.strip()}" (Probability: {chosen_token_prob:.2%})\n'
        output_string += "   Top 5 candidates for this position:\n"
        
        for j, (prob, index) in enumerate(zip(top_5_probs, top_5_indices)):
            decoded_token = tokenizer.decode(index)
            output_string += f"      {j+1}. \"{decoded_token.strip()}\" ({prob:.2%})\n"
        
        output_string += "\n"
        
    return output_string.strip()


@torch.inference_mode()
def extract_logits_first_step(
    model,
    tokenizer,
    prompt: str,
    target_tokens: List[str],
    device = 'cuda',
):
    """
    Greedily generates ONE token after *prompt* and returns the raw logits
    assigned to each token in *target_tokens* at that first generation step.

    Returns
    -------
    dict {token: logit}
    """
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Map each candidate token to a single ID
    token_id_map = {}
    for tok in target_tokens:
        ids = tokenizer(tok, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"'{tok}' is not a single-token string.")
        token_id_map[tok] = ids[0]

    # Generate exactly one new token (greedy)
    gen_out = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    first_step_logits = gen_out.scores[0][0]        # shape [vocab_size]

    # Extract logits for requested tokens
    return {tok: first_step_logits[tid].item() for tok, tid in token_id_map.items()}

# ---------- usage ----------
# prompt = "Answer with yes or no: Is acetaminophen mutagenic?\nA: "
# logits = extract_logits_first_step(model, tokenizer, prompt, [" yes", " no"])
# print(logits)          # {' yes': -3.21, ' no': -1.05}
# prediction = int(logits[" yes"] > logits[" no"])   # 1 = yes, 0 = no


class ClassificationAccuracyCallback(TrainerCallback):
    """
    A callback that evaluates classification accuracy or AUROC on a validation set
    at the end of each epoch. It assumes a binary classification task where the
    model's response is expected to contain 'yes' or 'no'.
    """
    def __init__(self, tokenizer, validation_dataset, inference_config, metric='accuracy', log_prefix="eval"):
        self.tokenizer = tokenizer
        self.validation_dataset = validation_dataset
        self.inference_config = inference_config
        self.metric = metric
        self.log_prefix = log_prefix
        self.history = []

        if self.metric not in ['accuracy', 'auroc']:
            raise ValueError("Metric must be either 'accuracy' or 'auroc'.")

    def on_epoch_end(self, args, state, control, model, **kwargs):
        model.eval()
        
        targets, preds = [], []
        
        for row in tqdm(self.validation_dataset, desc=f"Epoch {int(state.epoch)} evaluation"):
            prompt = row["text"]
            target = row["Y"]

            # Generate response from the model
            gen_text = generate_text(model, self.tokenizer, prompt, self.inference_config)
            generated_response = gen_text[len(prompt):].strip().lower()

            # Determine prediction
            if "yes" in generated_response:
                pred = 1
            elif "no" in generated_response:
                pred = 0
            else:
                # Fallback to logit comparison
                try:
                    # Note: Using " Yes" and " No" with a leading space is common for many tokenizers
                    probs = extract_logits_first_step(model, self.tokenizer, prompt, [" Yes", " No"])
                    pred = int(probs[" Yes"] > probs[" No"])
                except ValueError:
                    # If tokens are not single, default to a prediction (e.g., 0)
                    pred = 0

            targets.append(target)
            preds.append(pred)

        # Calculate metric
        if self.metric == "accuracy":
            score = accuracy_score(targets, preds)
        elif self.metric == "auroc":
            score = roc_auc_score(targets, preds)

        if state.is_world_process_zero:
            print(f"\nEpoch {int(state.epoch)} - {self.log_prefix}_{self.metric}: {score:.4f}")
            wandb.log({f"{self.log_prefix}/{self.metric}": score}, step=state.global_step)
        
        self.history.append({'step': state.global_step, 'epoch': state.epoch, f'{self.log_prefix}_{self.metric}': score})

        model.train()

    def get_results_as_dataframe(self):
        return pd.DataFrame(self.history)