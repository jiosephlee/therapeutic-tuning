import sys 
#sys.path.append('../../trl')

import math
import os
import wandb
import torch
from typing import Optional, List, Literal
import pandas as pd

# Third-party imports
from datasets import Dataset, load_dataset
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer
from utils.llm_configs import PeftConfig, ModelConfig, TrainingConfig, InferenceConfig

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
    
    assert(train_cfg.gradient_accumulation_steps <= len(text_chunks))
    log.info(f"[{tag}] Gradient_accumulation_steps ({train_cfg.gradient_accumulation_steps}) is less than {len(text_chunks)}")
    
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
    
    
class KnowledgeProbeCallback(TrainerCallback):
    """
    A unified callback that evaluates two types of perplexity on a custom set of
    knowledge probes in a single forward pass:
    1.  Whole statement perplexity: PPL over the entire probe text.
    2.  Targeted perplexity: PPL over the last three words of the probe.

    This avoids redundant model calls, logs both metrics to W&B, and
    stores them for external analysis.
    """
    def __init__(self, tokenizer: AutoTokenizer, probe_dataset_path: str, max_length: int, batch_size: int = 8, log_prefix="probe_ppl"):
        self.tokenizer = tokenizer
        self.log_prefix = log_prefix
        self.max_length = max_length
        self.batch_size = batch_size

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        df = pd.read_csv(probe_dataset_path)
        self.probes = df["raw_knowledge_statement"].tolist()
        self.sections = df["section"].tolist()
        self.probe_indices = df.index.tolist()

        # History for both metrics
        self.whole_history = []
        self.targeted_history = []

        # Pre-calculate context token lengths for targeted PPL
        self.context_token_lengths = []
        for statement in self.probes:
            words = statement.split()
            context_part = " ".join(words[:-3]) if len(words) > 3 else ""
            num_tokens = len(self.tokenizer(context_part, add_special_tokens=False)['input_ids'])
            self.context_token_lengths.append(num_tokens)

    def on_step_end(self, args, state, control, model, **kwargs):
        model.eval()
        device = model.device

        # Lists to store results from all mini-batches
        all_whole_perplexities = []
        all_targeted_perplexities = []
        
        # Process probes in mini-batches to conserve GPU memory
        for i in range(0, len(self.probes), self.batch_size):
            batch_probes = self.probes[i:i + self.batch_size]
            batch_context_lengths = self.context_token_lengths[i:i + self.batch_size]
            
            if not batch_probes:
                continue

            inputs = self.tokenizer(
                batch_probes,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                logits = outputs.logits
                
                # --- Common Calculations ---
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_attention_mask = attention_mask[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss_per_token = loss_per_token.view(input_ids.size(0), -1)

                # --- 1. Whole Statement Perplexity ---
                whole_masked_loss = loss_per_token * shift_attention_mask
                whole_sum_loss = whole_masked_loss.sum(dim=1)
                whole_num_tokens = shift_attention_mask.sum(dim=1)
                
                whole_non_zero_mask = whole_num_tokens > 0
                whole_mean_loss = torch.zeros_like(whole_sum_loss, dtype=torch.float32)
                valid_indices_whole = torch.where(whole_non_zero_mask)[0]
                if len(valid_indices_whole) > 0:
                    whole_mean_loss[valid_indices_whole] = whole_sum_loss[valid_indices_whole] / whole_num_tokens[valid_indices_whole]
                
                whole_perplexities = torch.exp(whole_mean_loss)
                all_whole_perplexities.append(whole_perplexities)

                # --- 2. Targeted (Last 3 Words) Perplexity ---
                target_mask = torch.zeros_like(shift_labels, dtype=torch.float)
                for j in range(len(batch_probes)): # Iterate over the mini-batch
                    # The loss for the N-th token is at index N-1 in shifted tensors
                    context_len = batch_context_lengths[j]
                    start_idx = max(0, context_len)
                    target_mask[j, start_idx:] = 1

                targeted_final_mask = shift_attention_mask * target_mask
                targeted_masked_loss = loss_per_token * targeted_final_mask
                targeted_sum_loss = targeted_masked_loss.sum(dim=1)
                targeted_num_tokens = targeted_final_mask.sum(dim=1)

                targeted_non_zero_mask = targeted_num_tokens > 0
                targeted_mean_loss = torch.zeros_like(targeted_sum_loss, dtype=torch.float32)
                valid_indices_targeted = torch.where(targeted_non_zero_mask)[0]
                if len(valid_indices_targeted) > 0:
                    targeted_mean_loss[valid_indices_targeted] = targeted_sum_loss[valid_indices_targeted] / targeted_num_tokens[valid_indices_targeted]

                targeted_perplexities = torch.exp(targeted_mean_loss)
                all_targeted_perplexities.append(targeted_perplexities)

        # Concatenate results from all batches
        whole_perplexities_all = torch.cat(all_whole_perplexities)
        targeted_perplexities_all = torch.cat(all_targeted_perplexities)

        # --- Logging and Storing ---
        if state.is_world_process_zero:
            log_data = {}
            
            # Create masks for valid (non-inf) perplexities
            valid_whole_mask = ~torch.isinf(whole_perplexities_all)
            valid_targeted_mask = ~torch.isinf(targeted_perplexities_all)

            # Log whole PPL avg
            if valid_whole_mask.any():
                avg_ppl = whole_perplexities_all[valid_whole_mask].mean().item()
                log_data[f"{self.log_prefix}/whole_avg"] = avg_ppl
            
            # Log targeted PPL avg
            if valid_targeted_mask.any():
                avg_ppl = targeted_perplexities_all[valid_targeted_mask].mean().item()
                log_data[f"{self.log_prefix}/targeted_avg"] = avg_ppl
            
            if log_data:
                wandb.log(log_data, step=state.global_step)
        
        # Store data internally
        self.whole_history.append({
            'step': state.global_step,
            'perplexities': whole_perplexities_all.cpu().tolist(),
        })
        self.targeted_history.append({
            'step': state.global_step,
            'perplexities': targeted_perplexities_all.cpu().tolist(),
        })
        
        model.train()

    def _get_dataframe_from_history(self, history):
        if not history:
            return pd.DataFrame()
        records = []
        for entry in history:
            step = entry['step']
            for i, perplexity in enumerate(entry['perplexities']):
                records.append({
                    'step': step,
                    'probe_index': self.probe_indices[i],
                    'section': self.sections[i],
                    'perplexity': perplexity
                })
        return pd.DataFrame(records)

    def get_whole_perplexity_dataframe(self):
        """
        Returns the collected whole-statement perplexity data as a pandas DataFrame.
        """
        return self._get_dataframe_from_history(self.whole_history)

    def get_targeted_perplexity_dataframe(self):
        """
        Returns the collected targeted (last three words) perplexity data as a pandas DataFrame.
        """
        return self._get_dataframe_from_history(self.targeted_history)


class CorpusPerplexityCallback(TrainerCallback):
    """
    Calculates the perplexity of an entire text corpus at the end of each
    training step using a strided sliding window approach. This provides a
    more accurate perplexity measure for long documents than naive chunking.
    Based on the Hugging Face documentation for PPL with fixed-length models.
    """
    def __init__(self, text_content: str, tokenizer: AutoTokenizer, max_length: int, stride: int = 512, log_prefix="corpus_perplexity"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.log_prefix = log_prefix
        self.encodings = self.tokenizer(text_content, return_tensors="pt")
        self.history = []

    def on_step_end(self, args, state, control, model, **kwargs):
        model.eval()
        device = model.device

        seq_len = self.encodings.input_ids.size(1)
        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            
            # Mask out tokens that are only used for context. The model will not
            # calculate loss for these tokens (label = -100).
            target_ids[:, :-trg_len] = -100

            if torch.all(target_ids == -100):
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
                continue

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                # outputs.loss is the *average* negative log-likelihood for the window.
                neg_log_likelihood = outputs.loss

            # To get the total NLL for the window, we multiply the average by the
            # number of tokens the loss was calculated over.
            num_valid_tokens = (target_ids != -100).sum().item()
            # The model internally shifts labels, so loss is on one less token per sequence.
            # Our batch size is 1 here.
            num_loss_tokens = num_valid_tokens - 1
            if num_loss_tokens > 0:
                nll_sum += neg_log_likelihood.item() * num_loss_tokens
                n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        if n_tokens > 0:
            avg_nll = nll_sum / n_tokens
            perplexity = torch.exp(torch.tensor(avg_nll))
        else:
            perplexity = torch.tensor(float('inf'))

        perplexity_item = perplexity.item()
        if state.is_world_process_zero:
            wandb.log({f"{self.log_prefix}/full_paper": perplexity_item}, step=state.global_step)
        
        self.history.append({'step': state.global_step, 'corpus_perplexity': perplexity_item})

        model.train()

    def get_results_as_dataframe(self):
        """
        Returns the collected corpus perplexity data as a pandas DataFrame.
        """
        return pd.DataFrame(self.history)


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