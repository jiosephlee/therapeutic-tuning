# add .. path 
import os
import sys
sys.path.append('../../')
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
    id="Qwen/Qwen2.5-7B",
    peft=llm_configs.PeftConfig(
        enabled=False,
        add_eot_token=False,  # No longer doing EOT token for LIMA
    ),
    quantization=llm_configs.QuantizationConfig(mode=None), # Use QLoRA
)

log.info("--- Configuration ---")
print(model_config.model_dump_json(indent=2))

log.info("\n--- Loading Model for Training ---")
model, tokenizer = llm_training.load_model_for_training(model_config, log, use_cpu_and_gpu=True)

def concat_columns(example, tokenizer):
    """
    Combine DOI/entity/fact/MolInfo/GeneInfo into one human-readable string.
    Empty or missing fields are omitted for that row.
    """

    chunks = []

    # 1) flat string columns
    if example.get("entity"):
        chunks.append(f"The following fact is for the entity '{example['entity']}'.")
    if example.get("fact"):
        chunks.append(f" {example['fact']}")

    # 2) MolInfo → [SMILES] …
    mol = example.get("MolInfo")
    if isinstance(mol, dict):
        smiles = mol.get("SMILES")
        if smiles:
            chunks.append(f"The SMILES string of this entity is '{smiles}'.")

    # # 3) GeneInfo → [GeneInfo] key: value, …
    # gene = example.get("GeneInfo")
    # if isinstance(gene, dict) and gene:
    #     def _fmt(key, val):
    #         return f'"{key}": {val}' if isinstance(val, int) else f'"{key}": "{val}"'
    #     fields = [_fmt(k, v) for k, v in gene.items() if v not in (None, "", [])]
    #     if fields:
    #         chunks.append(f"The NCBI Gene information of this entity is " + ", ".join(fields))
    #         print(f"The NCBI Gene information of this entity is " + ", ".join(fields))
    # join all parts with a single space
    return {"text": " ".join(chunks) + tokenizer.eos_token}

# ---- apply to your Dataset ----
# creates a new 'text' column, keeps the originals (remove_columns=[] by default)
ds_with_text = ds.map(concat_columns, fn_kwargs={"tokenizer": tokenizer},  desc="Building concatenated text")

medex_ds = ds_with_text.select_columns(["text"])

lima_training_config = llm_configs.TrainingConfig(
    run_name = "1M samples on medex/Qwen2.5-7B",
    num_train_epochs = 1,
    learning_rate  = 1e-5,
    logging_strategy = "steps", 
    logging_steps = 1,
    gradient_checkpointing=False,
    context_length = 512,
    use_liger_kernel=True,
    per_device_train_batch_size =1,
    gradient_accumulation_steps=128,
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
model.push_to_hub('jiosephlee/therapeutic_fine_tuning_Qwen-2.5-7B_1M_v2')
tokenizer.push_to_hub('jiosephlee/therapeutic_fine_tuning_Qwen-2.5-7B_1M_v2')