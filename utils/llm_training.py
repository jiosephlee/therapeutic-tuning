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
)
from trl import SFTConfig, SFTTrainer
from utils.llm_configs import PeftConfig, ModelConfig, TrainingConfig, InferenceConfig
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

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
    

class MedexKnowledgeProbeCallback(TrainerCallback):
    """
    A callback designed for the MEDEX dataset to evaluate perplexity on specific
    knowledge probes from a CSV file. It calculates perplexity for 'entity', 
    'text', and 'fact' columns, including "probe" versions for text and fact 
    that exclude initial words. It also provides plotting functionalities.
    """
    def __init__(self, tokenizer: AutoTokenizer, probe_dataset_path: str, max_length: int, batch_size: int = 8, log_prefix="medex_probe_ppl"):
        self.tokenizer = tokenizer
        self.log_prefix = log_prefix
        self.max_length = max_length
        self.batch_size = batch_size

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        df = pd.read_csv(probe_dataset_path)
        # Only store text and fact for perplexity calculation
        self.probes = {
            "text": df["text"].tolist(),
            "fact": df["fact"].tolist(),
        }
        self.probe_indices = df.index.tolist()

        # Calculate entity frequencies and assign buckets for analysis
        entity_counts = df['entity'].value_counts()
        self.entity_freq_map = df['entity'].map(entity_counts)
        bins = [0, 5, 10, 20, 30, 40, 50, 100, float('inf')]
        labels = ['1-5', '6-10', '11-20', '21-30', '31-40', '41-50', '51-100', '100+']
        self.entity_freq_buckets = pd.cut(self.entity_freq_map, bins=bins, labels=labels, right=True)

        self.history = []

        # Pre-calculate context token lengths for "probe" PPL
        self.context_token_lengths = {"text": [], "fact": []}
        for col in self.probes.keys(): # Will be ["text", "fact"]
            for statement in self.probes[col]:
                words = str(statement).split()
                # Determine number of words to use as context
                context_word_count = 5 if len(words) < 20 else 10
                context_part = " ".join(words[:context_word_count])
                num_tokens = len(self.tokenizer(context_part, add_special_tokens=False)['input_ids'])
                self.context_token_lengths[col].append(num_tokens)

    def on_step_end(self, args, state, control, model, **kwargs):
        model.eval()
        device = model.device

        step_results = {'step': state.global_step}
        log_data = {}

        probe_cols = ["text", "fact"] # Only calculate PPL for these columns
        for col in probe_cols:
            all_perplexities = []
            all_probe_perplexities = []
            
            # Process probes in mini-batches to conserve GPU memory
            for i in range(0, len(self.probes[col]), self.batch_size):
                batch_texts = [str(t) for t in self.probes[col][i:i + self.batch_size]]
                
                if not batch_texts:
                    continue

                inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
                ).to(device)

                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_attention_mask = attention_mask[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss_per_token = loss_per_token.view(input_ids.size(0), -1)

                # --- Whole Statement Perplexity ---
                whole_masked_loss = loss_per_token * shift_attention_mask
                whole_sum_loss = whole_masked_loss.sum(dim=1)
                whole_num_tokens = shift_attention_mask.sum(dim=1)
                
                whole_mean_loss = torch.zeros_like(whole_sum_loss, dtype=torch.float32)
                valid_indices_whole = torch.where(whole_num_tokens > 0)[0]
                if len(valid_indices_whole) > 0:
                    whole_mean_loss[valid_indices_whole] = whole_sum_loss[valid_indices_whole] / whole_num_tokens[valid_indices_whole]
                
                perplexities = torch.exp(whole_mean_loss)
                all_perplexities.append(perplexities)

                # --- "Probe" Perplexity ---
                batch_context_lengths = self.context_token_lengths[col][i:i + self.batch_size]
                target_mask = torch.zeros_like(shift_labels, dtype=torch.float)
                for j in range(len(batch_texts)):
                    start_idx = max(0, batch_context_lengths[j])
                    target_mask[j, start_idx:] = 1

                probe_final_mask = shift_attention_mask * target_mask
                probe_masked_loss = loss_per_token * probe_final_mask
                probe_sum_loss = probe_masked_loss.sum(dim=1)
                probe_num_tokens = probe_final_mask.sum(dim=1)

                probe_mean_loss = torch.zeros_like(probe_sum_loss, dtype=torch.float32)
                valid_indices_probe = torch.where(probe_num_tokens > 0)[0]
                if len(valid_indices_probe) > 0:
                    probe_mean_loss[valid_indices_probe] = probe_sum_loss[valid_indices_probe] / probe_num_tokens[valid_indices_probe]

                probe_perplexities = torch.exp(probe_mean_loss)
                all_probe_perplexities.append(probe_perplexities)

            # Consolidate and log results for the column
            final_perplexities = torch.cat(all_perplexities)
            step_results[f'{col}_perplexity'] = final_perplexities.cpu().tolist()
            log_data[f"{self.log_prefix}/{col}_avg_ppl"] = final_perplexities[~torch.isinf(final_perplexities)].mean().item()

            final_probe_perplexities = torch.cat(all_probe_perplexities)
            step_results[f'{col}_probe_perplexity'] = final_probe_perplexities.cpu().tolist()
            log_data[f"{self.log_prefix}/{col}_probe_avg_ppl"] = final_probe_perplexities[~torch.isinf(final_probe_perplexities)].mean().item()

        self.history.append(step_results)
        if state.is_world_process_zero:
            wandb.log(log_data, step=state.global_step)

        model.train()

    def get_results_dataframe(self):
        records = []
        for entry in self.history:
            step = entry['step']
            num_probes = len(self.probe_indices)
            for i in range(num_probes):
                record = {
                    'step': step,
                    'probe_index': self.probe_indices[i],
                    'entity_freq_bucket': self.entity_freq_buckets[i],
                }
                for col in ["text", "fact"]:
                    record[f'{col}_perplexity'] = entry[f'{col}_perplexity'][i]
                    record[f'{col}_probe_perplexity'] = entry[f'{col}_probe_perplexity'][i]
                records.append(record)
        return pd.DataFrame(records)

    def plot_average_perplexities(self, output_dir="."):
        df = self.get_results_dataframe()
        avg_df = df.groupby('step').mean(numeric_only=True)

        plt.figure(figsize=(12, 8))
        metrics_to_plot = ['text_perplexity', 'text_probe_perplexity', 'fact_perplexity', 'fact_probe_perplexity']
        for metric in metrics_to_plot:
            plt.plot(avg_df.index, avg_df[metric], label=metric)
        
        plt.title('Average Perplexity Over Training Steps')
        plt.xlabel('Training Step')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "average_perplexities.png"))
        plt.close()

    def plot_perplexity_by_entity_frequency(self, output_dir="."):
        df = self.get_results_dataframe()
        
        for probe_type in ['text_probe_perplexity', 'fact_probe_perplexity']:
            grouped_df = df.groupby(['entity_freq_bucket', 'step'])[probe_type].mean().reset_index()
            
            plt.figure(figsize=(14, 9))
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