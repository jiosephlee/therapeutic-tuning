import sys
import os
import logging
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
import torch # Assuming query_llm might need torch eventually
from importlib import reload

# Add project root to path if necessary (adjust based on your structure)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import query_llm
import utils.utils as utils
import utils.prompts as prompts # If needed for prompts
from utils.keys import WANDB_API_KEY # If needed for LLM API keys potentially

# --- Logging Setup ---
LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "create_long_context.log")
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logging.getLogger().handlers[1].setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
OUTPUT_DATASET_DIR = "./data/pubmedqa_long_context" # Directory to save the new dataset
BASE_DATASET_NAME = 'PubMedQA'
BASE_DATASET_SPLIT = 'train' # Or whichever split you want to process
DATASET_START_INDEX = 0
DATASET_END_INDEX = 50 # Process a small subset for testing first


# --- Dataset Processing Function ---
def generate_long_context(example):
    """
    Processes a single example from the dataset to generate long context.
    """
    question_id = example.get('id', 'unknown_id')
    logger.debug(f"Processing example ID: {question_id}")

    original_contexts = example['context']['contexts']
    original_question = example['question']

    # Combine original context into a single string
    combined_context = "\n".join(original_contexts)

    # Create the prompt for the LLM
    # You might want to refine this prompt significantly
    prompt = f"Please elaborate on the following text, making it significantly longer while preserving all key facts and ensuring the information remains relevant to the question: '{original_question}'. Do not add unrelated information. Focus on providing more detail, explanation, and background based *only* on the provided text.\n\nOriginal Text:\n{combined_context}\n\nElaborated Text:"

    # Call the (placeholder) LLM
    try:
        long_context = query_llm(prompt)
    except Exception as e:
        logger.error(f"Error querying LLM for example ID {question_id}: {e}")
        long_context = "[ERROR DURING LLM QUERY]" + "\n\n" + combined_context # Fallback

    # Return a new structure - keeping essential fields + new context
    return {
        "id": question_id,
        "question": original_question,
        "final_decision": example['final_decision'],
        # 'original_contexts': original_contexts, # Optional: keep original for reference
        "long_context": long_context # The new, elaborated context
    }

# --- Main Script ---
if __name__ == "__main__":
    logger.info(f"Loading base dataset: {BASE_DATASET_NAME}, split: {BASE_DATASET_SPLIT}")
    try:
        original_dataset = utils.load_dataset(
            BASE_DATASET_NAME,
            split=BASE_DATASET_SPLIT,
            start_index=DATASET_START_INDEX,
            end_index=DATASET_END_INDEX
        )
        logger.info(f"Loaded {len(original_dataset)} examples.")
    except Exception as e:
        logger.error(f"Failed to load base dataset: {e}")
        sys.exit(1)

    logger.info("Starting long context generation...")
    # Use map to process the dataset - might be slow if query_llm is slow and sequential
    # Consider adding num_proc > 1 if query_llm is thread-safe and I/O bound
    # Ensure your query_llm can handle potential parallel calls if num_proc > 1
    long_context_dataset = original_dataset.map(
        generate_long_context,
        batched=False, # Process one example at a time
        # num_proc=4, # Optional: Uncomment if query_llm is safe for parallel execution
    )
    logger.info("Finished generating long context.")

    # Remove original context if desired (optional, depends on generate_long_context)
    # long_context_dataset = long_context_dataset.remove_columns(['context'])

    logger.info(f"Saving processed dataset to: {OUTPUT_DATASET_DIR}")
    try:
        # Save as a Dataset object (Arrow format)
        long_context_dataset.save_to_disk(OUTPUT_DATASET_DIR)

        # Optional: Save as JSON Lines as well (useful for inspection)
        output_jsonl_path = os.path.join(OUTPUT_DATASET_DIR, f"{BASE_DATASET_SPLIT}.jsonl")
        long_context_dataset.to_json(output_jsonl_path, orient="records", lines=True)

        logger.info("Dataset saved successfully.")
        logger.info(f"Output structure example: {long_context_dataset[0]}") # Log first example structure

    except Exception as e:
        logger.error(f"Failed to save processed dataset: {e}")

    logger.info("Script finished.") 