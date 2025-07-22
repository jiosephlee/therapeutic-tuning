# add .. path 
import os
import sys
sys.path.append('..')
import utils.llm_training as llm_training
import utils.llm_configs as llm_configs

import logging

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

os.environ["WANDB_PROJECT"]="medex_continued_pretraining"

from datasets import load_dataset

ds = load_dataset("medexanon/Medex")['train'].select(range(1000000))

# === Cell 1: Configuration ===
model_config = llm_configs.ModelConfig(
    id="Qwen/Qwen2.5-0.5B",
    peft=llm_configs.PeftConfig(
        enabled=False,
        add_eot_token=False,  # No longer doing EOT token for LIMA
    ),
    quantization=llm_configs.QuantizationConfig(mode=None), # Use QLoRA
)

log.info("--- Configuration ---")
print(model_config.model_dump_json(indent=2))

log.info("\n--- Loading Model for Training ---")
model, tokenizer = llm_training.load_model_for_training(model_config, log, use_cpu_and_gpu=False)

def concat_columns_and_explode(batch, tokenizer):
    """
    Processes a batch of examples to generate a list of text entries. 'fact'
    becomes one entry, and the SMILES/entity association becomes another.
    Since this runs in batched mode, the output list can be longer than the
    input batch, effectively creating new rows.
    """
    all_texts = []
    # The input 'batch' is a dictionary of lists (e.g., batch['fact'] is a list of facts)
    for i in range(len(batch["fact"])):
        # 1) Create a row for the fact text, if it exists
        fact = batch["fact"][i]
        if fact:
            all_texts.append(f"{fact.strip()}{tokenizer.eos_token}")

        # 2) Create a row for the SMILES/entity association
        mol = batch["MolInfo"][i]
        if isinstance(mol, dict):
            smiles = mol.get("SMILES")
            entity = batch["entity"][i]
            if smiles and entity:
                smiles_text = f"The SMILES string of '{entity}' is '{smiles}'."
                all_texts.append(f"{smiles_text}{tokenizer.eos_token}")

    # Return a dictionary where the 'text' key maps to the list of all generated strings
    return {"text": all_texts}

# ---- apply to your Dataset ----
# Creates a new 'text' column and explodes list items into new rows
ds_with_text = ds.map(
    concat_columns_and_explode,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=ds.column_names, 
    batched=True,
    desc="Building and exploding text rows"
)

# Shuffle the newly created dataset and then select only the 'text' column
medex_ds = ds_with_text.shuffle(seed=42).select_columns(["text"])

lima_training_config = llm_configs.TrainingConfig(
    run_name = "1M samples on medex (prompt ablation 3)",
    num_train_epochs = 1,
    learning_rate  = 1e-5,
    logging_strategy = "steps", 
    logging_steps = 1,
    gradient_checkpointing=False,
    context_length = 512,
    use_liger_kernel=True,
    per_device_train_batch_size =8,
    gradient_accumulation_steps=16,
    # warmup_steps  = 0, # LIMA specifies no warmup, so we set this explicitly
    warmup_ratio = 0.3, # Use our default warmup ratio instead
    packing=True,
    padding_free = True,
    sequential_sampling = False,
    reverse_ffd_packing= False,
    remove_unused_columns=False,
)


# === Run LIMA Fine-Tuning ===
log.info("\n--- Starting LIMA Fine-Tuning ---")
# The model object will be updated with the fine-tuned weights
trainer = llm_training.sft_train_on_dataset(
    model=model,
    tokenizer=tokenizer,
    log=log,
    train_dataset=medex_ds,
    train_cfg=lima_training_config,
    train=True,
    use_liger_loss = True
)

# Save model before we LIMA tune
model.push_to_hub('jiosephlee/therapeutic_fine_tuning_1M_v3')
tokenizer.push_to_hub('jiosephlee/therapeutic_fine_tuning_1M_v3')