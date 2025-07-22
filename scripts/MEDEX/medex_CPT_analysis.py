# add .. path 
import os
import sys
sys.path.append('../..')
import utils.llm_training as llm_training
import utils.llm_configs as llm_configs
import wandb
import logging

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

run_name = "Analysis on 10K Facts"

run = wandb.init(
    project="medex_continued_pretraining",
    name=run_name,
    group="Analysis",
)
from datasets import load_dataset

ds = load_dataset("medexanon/Medex", split="train[:1%]").select(range(10000))

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
        # Create the full probe text matching create_probe_text format
        chunks = []
        
        entity = batch["entity"][i]
        fact_text = batch["fact"][i]
        
        # 1. Entity
        if entity:
            chunks.append(f"Entity: {entity}")
        
        # 2. SMILES
        mol = batch["MolInfo"][i]
        if isinstance(mol, dict):
            smiles = mol.get("SMILES")
            if smiles:
                chunks.append(f"SMILES: {smiles}")
        
        # 3. Fact
        if fact_text:
            chunks.append(fact_text)
        
        # Join with ". " and add a final period, then EOS token
        if chunks:
            full_text = ". ".join(chunks)
            if not full_text.endswith('.'):
                full_text += '.'
            all_texts.append(f"{full_text}{tokenizer.eos_token}")

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
    run_name = run_name,
    num_train_epochs = 10,
    learning_rate  = 1e-5,
    logging_strategy = "steps", 
    logging_steps = 1,
    gradient_checkpointing=False,
    context_length = 1024,
    use_liger_kernel=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=64,
    # warmup_steps  = 0, # LIMA specifies no warmup, so we set this explicitly
    warmup_ratio = 0.3, # Use our default warmup ratio instead
    packing=True,
    padding_free = True,
    sequential_sampling = False,
    reverse_ffd_packing= False,
    remove_unused_columns=False,
)


# === Create Callback ===
log.info("\n--- Initializing Knowledge Probe Callback ---")
knowledge_probe_callback = llm_training.MedexKnowledgeProbeCallback(
    tokenizer=tokenizer,
    probe_dataset_path="../../data/MEDEX/knowledge_probes_10000.csv",
    max_length=512, # Should match context_length
    batch_size=16
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
    use_liger_loss = True,
    callbacks=[knowledge_probe_callback]
)

# === Plotting Results ===
log.info("\n--- Generating Plots ---")
output_plot_dir = f"results/plots/{lima_training_config.run_name}"
knowledge_probe_callback.plot_average_perplexities(output_dir=output_plot_dir)
knowledge_probe_callback.plot_perplexity_by_entity_frequency(output_dir=output_plot_dir)
log.info(f"Plots saved to {output_plot_dir}")
