# add .. path 
import os
import sys
sys.path.append('..')
import utils.llm_training as llm_training
import utils.llm_configs as llm_configs
import wandb
import logging
import re
from tqdm import tqdm
import numpy as np
from datasets import Dataset
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score

# --- Basic Configuration ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="AMES")
parser.add_argument("--metric", type=str, default="auroc")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
_METHOD = 'text'
# Model names: jiosephlee/therapeutic_fine_tuning_1M_v2, jiosephlee/therapeutic_fine_tuning_10M, jiosephlee/therapeutic_fine_tuning_36M
args = parser.parse_args()
run_name = f"{args.dataset}_fine_tuning/{args.model}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

run = wandb.init(
    project="medex_fine_tuning",
    name=run_name,
    tags=[args.dataset],
    group="Finetuning",
)

# --- Load Data and Preprocess---
train_df = pd.read_csv(f'./../data/TDC/{args.dataset}/train_df.csv')
val_df = pd.read_csv(f'./../data/TDC/{args.dataset}/val_df.csv')
test_df = pd.read_csv(f'./../data/TDC/{args.dataset}/test_df.csv')

def row_to_text( row, split='train', dataset='AMES_1'):
    if dataset == 'AMES':
        text = f"SMILES: {row['Drug']}\nQuestion: Is the drug represented by this SMILES string mutagenic?\nAnswer:"
        if split == 'train':
            text += f"{' Yes, the drug is mutagenic.' if row['Y']==1 else ' No, the drug is not mutagenic.'}"
    if dataset == 'AMES_1':
        text = f"SMILES: {row['Drug']}\nQuestion: Is the drug represented by this SMILES string mutagenic?\nAnswer:"
        if split == 'train':
            text += f"{' Yes, the drug is mutagenic.' if row['Y']==1 else ' No, the drug is not mutagenic.'}"
    if dataset == 'AMES_2':
        text = f"SMILES: {row['Drug']}\nQuestion: Is the drug represented by this SMILES string mutagenic?\nAnswer:"
        if split == 'train':
            text += f"{' Yes.' if row['Y']==1 else ' No.'}"
    if dataset == 'AMES_3':
        text = f"Question: Is the drug represented by this SMILES string, {row['Drug']}, mutagenic?\nAnswer:"
        if split == 'train':
            text += f"{' Yes, the drug is mutagenic.' if row['Y']==1 else ' No, the drug is not mutagenic.'}"
    elif dataset == 'Skin':
        text = f"SMILES: {row['Drug']}\nQuestion: Can the drug represented by this SMILES string cause skin reaction?\nAnswer:"
        if split == 'train':
            text += f"{' Yes, the drug can cause skin reaction.' if row['Y']==1 else ' No, the drug cannot cause skin reaction.'}"
    elif dataset == 'Carcinogens':
        text = f"SMILES: {row['Drug']}\nQuestion: Is the drug represented by this SMILES string carcinogenic?\nAnswer:"
        if split == 'train':
            text += f"{' Yes, the drug is carcinogenic.' if row['Y']==1 else ' No, the drug is not carcinogenic.'}"
    if dataset == 'BBB':
        text = f"SMILES: {row['Drug']}\nQuestion: Can the drug represented by this SMILES string penetrate the blood-brain barrier to deliver to the site of action?\nAnswer:"
        if split == 'train':
            text += f"{' Yes, the drug can penetrate the blood-brain barrier.' if row['Y']==1 else ' No, the drug cannot penetrate the blood-brain barrier.'}"
    if dataset == 'Bioavailability':
        text = f"SMILES: {row['Drug']}\nQuestion: Can the drug represented by this SMILES string be absorbed into the bloodstream?\nAnswer:"
        if split == 'train':
            text += f"{' Yes, the drug can be absorbed into the bloodstream.' if row['Y']==1 else ' No, the drug cannot be absorbed into the bloodstream.'}"
    if dataset == 'hERG':
        text = f"SMILES: {row['Drug']}\nQuestion: Does the drug represented by this SMILES string block the hERG channel (IC50 < 10uM)?\nAnswer:"
        if split == 'train':
            text += f"{' Yes, the drug blocks the hERG channel.' if row['Y']==1 else ' No, the drug does not block the hERG channel.'}"
    return text

def row_to_prompt( row, dataset='AMES'):
    if dataset == 'AMES':
        prompt = f"SMILES: {row['Drug']}\nQuestion: Is the drug represented by this SMILES string mutagenic?\nAnswer:"
    elif dataset == 'Skin Reaction':
        prompt = f"Q: This is the SMILES string of the drug: {row['Drug']}. Can this drug cause skin reaction?\nA: "

    return prompt

def row_to_completion( row, dataset='AMES'):
    if dataset == 'AMES':
        completion = " Yes, the drug is mutagenic."
    elif dataset == 'Skin Reaction':
        completion = f"Question: This is the SMILES string of the drug: {row['Drug']}. Can this drug cause skin reaction?\nA: "
    return completion

def transform_df(train_df, val_df, test_df, dataset, method='text'):
    if method == 'text': 
        train_df["text"] = train_df.apply(row_to_text, axis=1, split = 'train', dataset = dataset)
    elif method=='completion':
        train_df["prompt"] = train_df.apply(row_to_prompt, axis=1, dataset = dataset)
        train_df["completion"] = train_df.apply(row_to_completion, axis=1, dataset = dataset)
    val_df["text"] = val_df.apply(row_to_text, axis=1, split = 'val', dataset = dataset)
    test_df["text"] = test_df.apply(row_to_text, axis=1, split = 'test', dataset = dataset)

transform_df(train_df, val_df, test_df, args.dataset, method=_METHOD)

training_ds = Dataset.from_pandas(train_df, preserve_index=False)
training_ds = training_ds.select_columns(
                    {"text", "Y", "prompt", "completion"}.intersection(training_ds.column_names)
                )
val_ds = Dataset.from_pandas(val_df, preserve_index=False)
val_ds = val_ds.select_columns(
                    {"text", "Y", "prompt", "completion"}.intersection(val_ds.column_names)
                )
test_ds = Dataset.from_pandas(test_df, preserve_index=False)
test_ds = test_ds.select_columns(
                    {"text", "Y", "prompt", "completion"}.intersection(test_ds.column_names)
                )

log.info(f"Training dataset example: {training_ds[0]}")
log.info(f"Validation dataset example: {val_ds[0]}")
log.info(f"Test dataset example: {test_ds[0]}")

# --- Load Model ---
model_config = llm_configs.ModelConfig(
    id=args.model,
    peft=llm_configs.PeftConfig(
        enabled=False,
        add_eot_token=False,  # No longer doing EOT token for LIMA
    ),
    quantization=llm_configs.QuantizationConfig(mode=None), # Use QLoRA
)

log.info("--- Model Configuration ---")
log.info(model_config.model_dump_json(indent=2))

log.info("\n--- Loading Model for Training ---\n")
model, tokenizer = llm_training.load_model_for_training(model_config, log)

lima_training_config = llm_configs.TrainingConfig(
    run_name = run_name,
    num_train_epochs = 1,
    learning_rate  = 4e-5,
    logging_strategy = "steps", 
    logging_steps = 1,
    gradient_checkpointing=False,
    context_length = 4096,
    use_liger_kernel=True,
    per_device_train_batch_size = 128,
    gradient_accumulation_steps=1,
    warmup_steps  = 0, # If 0, it does not override warmup ratio
    warmup_ratio = 0.1, # Use our default warmup ratio instead
    packing = False,
    padding_free = True,
    sequential_sampling = False,
    reverse_ffd_packing= False,
    remove_unused_columns=False,
)

# --- Train ---
log.info(f"\n--- Starting {args.dataset} Fine-Tuning ---")
trainer = llm_training.sft_train_on_dataset(
    model=model,
    tokenizer=tokenizer,
    log=log,
    train_dataset=training_ds,
    train_cfg=lima_training_config,
    train=True,
    use_liger_loss = True
)

log.info("\n\n--- Fine-Tuning Complete ---\n\n")
log.info(f"Training arguments: {trainer.args}")

# --- Evaluate ---
inference_cfg = llm_configs.InferenceConfig(
    temperature=0,
    do_sample=False,
    repetition_penalty=1.0,
    max_new_tokens=64,
)

targets, preds = [], []

for i in tqdm(range(len(val_ds)), desc="Inference on validation set"):
    row = val_ds[i]
    prompt = row["text"]
    gt_answer = "yes" if row["Y"] == 1 else "no"
    
    gen_text = llm_training.generate_text(model, tokenizer, prompt, inference_cfg)
    
    # Extract generated text (remove the prompt part)
    generated_response = gen_text[len(prompt):].strip().lower()

    if i < 10:
        # print(f"Prompt: {prompt}")
        print(f"Generated response: {gen_text}")
        print(f"GT answer: {gt_answer}")
        print("-"*100)
    # Simple matching - check if "yes" or "no" appears in the response
    if "yes" in generated_response:
        pred_answer = 1
    elif "no" in generated_response:
        pred_answer = 0
    else:
        probs = llm_training.extract_logits_first_step(model, tokenizer, prompt, ["Yes","No"])
        pred_answer = int(probs["Yes"] > probs["No"]) 

    targets.append(gt_answer)
    preds.append(pred_answer)

targets = np.array(targets)
preds = np.array(preds)

if args.metric == "accuracy":
    accuracy = np.mean(targets == preds)
    print(f"\nAccuracy on {len(targets)} examples: {accuracy:.4f}")
elif args.metric == "auroc":
    auroc = roc_auc_score(targets, preds)
    print(f"\nAUROC on {len(targets)} examples: {auroc:.4f}")


# --- Evaluate ---
inference_cfg = llm_configs.InferenceConfig(
    temperature=0,
    do_sample=False,
    repetition_penalty=1.0,
    max_new_tokens=64,
)

targets, preds = [], []

for i in tqdm(range(len(test_ds)), desc="Inference on validation set"):
    row = test_ds[i]
    prompt = row["text"]
    gt_answer = "yes" if row["Y"] == 1 else "no"
    
    gen_text = llm_training.generate_text(model, tokenizer, prompt, inference_cfg)
    
    # Extract generated text (remove the prompt part)
    generated_response = gen_text[len(prompt):].strip().lower()

    if i < 10:
        # print(f"Prompt: {prompt}")
        print(f"Generated response: {gen_text}")
        print(f"GT answer: {gt_answer}")
        print("-"*100)
    # Simple matching - check if "yes" or "no" appears in the response
    if "yes" in generated_response:
        pred_answer = 1
    elif "no" in generated_response:
        pred_answer = 0
    else:
        probs = llm_training.extract_logits_first_step(model, tokenizer, prompt, ["Yes","No"])
        pred_answer = int(probs["Yes"] > probs["No"]) 
        # If neither yes nor no is found, skip this example
        # continue
    
    targets.append(gt_answer)
    preds.append(pred_answer)

targets = np.array(targets)
preds = np.array(preds)

if args.metric == "accuracy":
    accuracy = np.mean(targets == preds)
    print(f"\nAccuracy on {len(targets)} examples: {accuracy:.4f}")
elif args.metric == "auroc":
    auroc = roc_auc_score(targets, preds)
    print(f"\nAUROC on {len(targets)} examples: {auroc:.4f}")