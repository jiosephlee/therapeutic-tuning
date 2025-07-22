from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
import torch
import re
from tqdm import tqdm
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
import copy # Needed for deep copying state dict

import sys
import os
sys.path.append(os.path.abspath(".."))
from importlib import reload
import utils.utils as utils
import utils.prompts as prompts
from utils.keys import WANDB_API_KEY
reload(utils)
reload(prompts)

# Track experiment
import wandb
wandb.login(key=WANDB_API_KEY) 
os.environ["WANDB_PROJECT"]="Fine-Tuning-or-Retrieval"

# --- Logging Setup ---
LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "experiment.log")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
import logging # Add logging import
logging.basicConfig(
    level=logging.DEBUG, # Capture debug messages and above
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'), # Write to file (overwrite mode)
        logging.StreamHandler() # Write to console
    ]
)
# Set console handler level to INFO to reduce console verbosity
logging.getLogger().handlers[1].setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Configuration ---
max_seq_length = 2048
dtype = None
load_in_4bit = True
model_name = "unsloth/Meta-Llama-3.1-8B" # Or your preferred model
original_seed = 3407 # Define the base seed

logger.info("Loading base model...") # Replace print
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
logger.info("Base model loaded.") # Replace print

EOS_TOKEN = tokenizer.eos_token
if tokenizer.pad_token is None:
    logger.info("Setting pad token to EOS token.") # Replace print
    tokenizer.pad_token = tokenizer.eos_token

logger.info("Adding LoRA adapters...") # Replace print
model = FastLanguageModel.get_peft_model(
    base_model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head",],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=original_seed, # Use base seed here
    use_rslora=True,
    loftq_config=None,
)
logger.info("LoRA adapters added.") # Replace print

# --- Dataset Loading ---
logger.info("Loading dataset...") # Replace print
dataset = utils.load_dataset('PubMedQA', split='train', start_index=0, end_index=50) # Use 'train', smaller subset first
logger.info(f"Dataset loaded with {len(dataset)} examples.") # Replace print

# --- Helper Functions ---

def format_pretraining_text(context_list):
    """Formats context list into a single string for pre-training."""
    return "\n".join(context_list) + EOS_TOKEN

def format_qa_prompt(background, question):
    """Formats background and question into the Yes/No prompt."""
    return f"Background: {background}\n\nQuestion: {question}\n\nPlease answer with Yes or No." # EOS is handled by generation

def parse_yes_no(text):
    """Parses generated text to extract 'yes' or 'no'."""
    text_lower = text.lower().strip()
    # More robust parsing
    if re.search(r"^\s*yes", text_lower):
        return "yes"
    elif re.search(r"^\s*no", text_lower):
        return "no"
    # Fallback if not at the beginning
    elif "yes" in text_lower:
        return "yes"
    elif "no" in text_lower:
        return "no"
    return "unknown"

def evaluate_question(model, tokenizer, qa_prompt_text):
    """Generates an answer for the QA prompt and parses Yes/No."""
    logger.debug(f"Evaluating QA prompt: {qa_prompt_text[:100]}...") # Debug log
    FastLanguageModel.for_inference(model) # <<< Enable fast inference
    # model.eval() # Trainer should handle this
    # Note: No EOS token added in format_qa_prompt now, let generation handle it
    inputs = tokenizer(qa_prompt_text,
                       return_tensors="pt",
    ).to(model.device) # Leave space

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        use_cache=True
    )
    # Decode only the generated part
    # Input length: inputs['input_ids'].shape[1]
    prediction_text = tokenizer.batch_decode(outputs)[0]
    logger.debug(f"Raw prediction: '{prediction_text}'") # Debug log raw output
    parsed_answer = parse_yes_no(prediction_text)
    logger.debug(f"Parsed answer: '{parsed_answer}'") # Debug log parsed output
    # Optional: Put model back into training mode if needed outside this function
    # model.train() # Typically trainer handles this before training step
    return parsed_answer

# --- Experiment Setup ---
num_finetune_epochs_per_question = 5
results = []

# --- Main Experiment Loop ---
logger.info("Starting experiment loop...") # Replace print
for idx, example in enumerate(tqdm(dataset, desc="Processing Questions")):
    question_id = example.get('id', f'idx_{idx}')
    true_answer = example['final_decision'].lower()
    question_text = example['question']
    contexts = example['context']['contexts'] # Ensure 'contexts' exists
    background_text = "\n".join(contexts)

    logger.debug(f"Processing Question ID: {question_id}") # Debug log

    current_results = {
        'id': question_id,
        'question': question_text,
        'true_answer': true_answer,
        'predictions': {}
    }

    qa_prompt_text = format_qa_prompt(background_text, question_text)

    # 1. Pre-Tune Evaluation
    logger.debug(f"[{question_id}] Resetting model and performing pre-tune evaluation.") # Debug log
    model = FastLanguageModel.get_peft_model(
    base_model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head",],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=original_seed, # Use base seed here
    use_rslora=True,
    loftq_config=None,
    )
    pre_train_pred = evaluate_question(model, tokenizer, qa_prompt_text)
    current_results['predictions']['pre_train'] = pre_train_pred
    logger.debug(f"[{question_id}] Pre-train Prediction: {pre_train_pred} (True: {true_answer})") # Debug log

    # 2. Prepare Context Data
    logger.debug(f"[{question_id}] Preparing context data for tuning.") # Debug log
    context_for_tuning = format_pretraining_text(contexts)
    tuning_data = Dataset.from_dict({"text": [context_for_tuning]})

    # 3. Iterative Fine-Tuning & Evaluation Loop
    for epoch in range(1, num_finetune_epochs_per_question + 1):
        logger.debug(f"[{question_id}] Starting fine-tuning epoch {epoch}/{num_finetune_epochs_per_question}.") # Debug log
        # Configure trainer - Use original args where possible
        temp_output_dir = f"./outputs_temp_{question_id}_epoch{epoch}"

        # Define arguments, keeping originals where feasible
        args = UnslothTrainingArguments(
            # --- Args to keep from original (potentially) ---
            warmup_ratio = 0.1,           # Original: 0.1
            learning_rate = 5e-5,         # Original: 5e-5
            # embedding_learning_rate = 5e-6, # Original: 5e-6 (can include if needed)
            optim = "adamw_8bit",         # Original: adamw_8bit
            weight_decay = 0.00,          # Original: 0.00
            lr_scheduler_type = "cosine", # Original: cosine (though effect minimal for 1 step)
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            # --- Args specific to this step ---
            per_device_train_batch_size = 1, # MUST be 1 for single example
            gradient_accumulation_steps = 1, # MUST be 1 for single step update
            num_train_epochs = 1,          # MUST be 1 for single step update
            logging_steps = 10,            # Adjust logging frequency if desired (original was 1)
            seed = original_seed ,  # Vary seed per step
            output_dir = temp_output_dir,  # Temporary output
            report_to = "wandb" if "WANDB_PROJECT" in os.environ else "none", # Report to wandb if configured
            save_strategy = "no",          # Disable saving checkpoints
        )

        trainer = UnslothTrainer(
            model=model, # Pass current model state
            tokenizer=tokenizer,
            train_dataset=tuning_data,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=1,
            args=args, # Use the defined args
        )

        # Fine-tune for one step
        logger.debug(f"[{question_id}][Epoch {epoch}] Starting trainer.train()") # Debug log
        # Trainer should handle model.train() / model.eval() transitions
        train_result = trainer.train()
        # Check if training_loss is available
        training_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else "N/A"
        logger.debug(f"[{question_id}][Epoch {epoch}] Training finished. Loss: {training_loss}") # Debug log loss

        # Evaluate on the question *after* this epoch
        logger.debug(f"[{question_id}][Epoch {epoch}] Evaluating question post-tuning.") # Debug log
        epoch_pred = evaluate_question(model, tokenizer, qa_prompt_text)
        current_results['predictions'][f'epoch_{epoch}'] = epoch_pred
        logger.debug(f"[{question_id}][Epoch {epoch}] Prediction: {epoch_pred} (True: {true_answer})") # Debug log

        # Optional cleanup
        import shutil
        if os.path.exists(temp_output_dir):
             logger.debug(f"[{question_id}][Epoch {epoch}] Cleaning up temporary directory: {temp_output_dir}") # Debug log
             shutil.rmtree(temp_output_dir)

    results.append(current_results)
    logger.debug(f"Finished processing Question ID: {question_id}") # Debug log

logger.info("Experiment loop finished.") # Replace print

# --- Analysis ---
logger.info("--- Analyzing Results ---") # Replace print
if not results:
    logger.warning("No results collected.") # Use warning level
else:
    df = pd.DataFrame(results)
    try:
        predictions_df = pd.json_normalize(df['predictions'])
        analysis_df = pd.concat([df[['id', 'question', 'true_answer']], predictions_df], axis=1)
    except Exception as e:
        logger.error(f"Error processing results into DataFrame: {e}") # Log error
        analysis_df = pd.DataFrame() # Create empty df to avoid further errors

    if not analysis_df.empty:
        from sklearn.metrics import accuracy_score
        accuracies = {}
        stages = ['pre_train'] + [f'epoch_{e}' for e in range(1, num_finetune_epochs_per_question + 1)]

        for stage in stages:
            if stage in analysis_df.columns:
                valid_preds_mask = analysis_df[stage] != 'unknown'
                # Ensure true_answer column exists and has data before calculating accuracy
                if 'true_answer' in analysis_df.columns and not analysis_df['true_answer'].isnull().all():
                    accuracy = accuracy_score(
                        analysis_df.loc[valid_preds_mask, 'true_answer'],
                        analysis_df.loc[valid_preds_mask, stage]
                    ) if valid_preds_mask.sum() > 0 else 0.0 # Handle case with zero valid preds
                else:
                    accuracy = 0.0 # Cannot calculate accuracy if true answers are missing
                    logger.warning(f"Cannot calculate accuracy for stage '{stage}' due to missing true answers.")

                num_unknown = len(analysis_df) - valid_preds_mask.sum()
                accuracies[stage] = (accuracy, num_unknown)
            else:
                logger.warning(f"Stage '{stage}' not found in results columns.") # Log warning
                accuracies[stage] = (0.0, len(analysis_df))

        logger.info(f"Processed {len(df)} questions.") # Replace print
        logger.info("Accuracies (Ignoring 'unknown' predictions):") # Replace print
        for stage, (acc, unknown_count) in accuracies.items():
            total_count = len(analysis_df)
            valid_count = total_count - unknown_count
            logger.info(f"- {stage}: {acc:.4f} ({valid_count}/{total_count} valid predictions, {unknown_count} unknown)") # Replace print

        output_filename = "rq1_experiment_results.csv"
        try:
            analysis_df.to_csv(output_filename, index=False)
            logger.info(f"Detailed results saved to {output_filename}") # Replace print
        except Exception as e:
            logger.error(f"Failed to save results to CSV: {e}") # Log error
    else:
        logger.error("Analysis DataFrame is empty, skipping accuracy calculation and saving.")