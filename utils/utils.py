import json
import pandas as pd
import datasets
import os
import time
from openai import OpenAI
from utils.keys import OPENAI_API_KEY, DATABRICKS_TOKEN


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

client_safe = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://adb-4750903324350629.9.azuredatabricks.net/serving-endpoints"
)

def save_predictions():
    pass

def evaluate_metrics():
    pass

#########################
## For Dataset Processing
#########################

_PROJECT_PATH = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval"

_DATASETS = {
    "MedQA": 
        {'test_set_filepath': f"{_PROJECT_PATH}/data/MedQA/questions/US/4_options/phrases_no_exclude_test.jsonl",
        'format': 'jsonl',
        },
    "BioASQ": 
        {'test_set_filepath': "./data/mimic-iv-public/triage_counterfactual.csv",
        'format': 'csv',
        'target': 'acuity',
        'training_set_filepath':'./data/mimic-iv-public/triage_public.csv',
        },
    "PubMedQA": 
        {'dataset_name': "qiaojin/PubMedQA",
         'train_set_filepath': "pqa_artificial",
         'test_set_filepath': "pqa_labeled",
        'format': 'huggingface',
        },
    }

def save_metrics(metrics,  filename):
    output_file = f"{_PROJECT_PATH}/results/metrics/{filename}_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
             
def load_dataset(dataset, start_index=None, end_index=None, split='test'):
    if dataset not in _DATASETS:
        raise ValueError("Dataset not found in _DATASETS.")
    if _DATASETS[dataset]['format'] == 'huggingface':
        path = _DATASETS[dataset][f'{split}_set_filepath'] 
        data = datasets.load_dataset(_DATASETS[dataset]['dataset_name'], path)['train']
        print(data)
        
        if start_index is not None and end_index is not None:
            # Using slicing instead of select method
            data = data.select(range(start_index, end_index))
    else:
        filepath = _DATASETS[dataset]['test_set_filepath']
        format = _DATASETS[dataset]['format']
        if format == 'jsonl':
            data = load_jsonl(filepath, start_index, end_index)
        elif format == 'csv':
            data = pd.read_csv(filepath).loc[start_index:end_index]
        else:
            raise ValueError(f"Unsupported format: {format}")
    return data
    
def load_jsonl(filepath, start_index, end_index):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for i, line in enumerate(f) if i <= end_index and i >= start_index]
    return data

def load_predictions(filename, format='txt', save_path=f"{_PROJECT_PATH}/results/predictions"):
    if format == 'csv':
        filename = f"{save_path}/{filename}.csv"
        predictions = pd.read_csv(filename)
    else: 
        filename = f"{save_path}/{filename}.txt"
        with open(filename, 'r') as f:
            predictions = [json.loads(line.strip()) for line in f]
    return predictions

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()


def query_llm(prompt, max_tokens=1000, temperature=0, top_p=0, max_try_num=10, model="gpt-4o-mini", debug=False, return_json=False, json_schema=None, logprobs=False, system_prompt_included=True, is_hippa=False):
    if debug:
        if system_prompt_included:
            print(f"System prompt: {prompt['system']}")
            print(f"User prompt: {prompt['user']}")
        else:
            print(prompt)
        print(f"Model: {model}")
    if is_hippa and ('gpt' not in model and 'o3' not in model):
        raise ValueError("HIPPA compliance requires GPT models")
    curr_try_num = 0
    while curr_try_num < max_try_num:
        try:
            if 'gpt' in model or 'o3' in model or 'o1' in model or 'o4' in model:
                response = query_gpt(prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, return_json=return_json, json_schema=json_schema, logprobs=logprobs, system_prompt_included=system_prompt_included, is_hippa=is_hippa, debug=debug)
                if logprobs:
                    return response.choices[0].message.content.strip(), response.choices[0].logprobs
            return response
        except Exception as e:
            raise e
        # except Exception as e:
        #     if 'gpt' in model:
        #         print(f"Error making OpenAI API call: {e}")
        #     else: 
        #         print(f"Error making API call: {e}")
        #     curr_try_num += 1
        #     time.sleep(1)
        #     if curr_try_num >= 3 and return_json:
        #         response = query_llm(prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, return_json=False, json_schema=json_schema, logprobs=logprobs, system_prompt_included=system_prompt_included, is_hippa=is_hippa, debug=debug)
        #         prompt=f"""Turn the following text into a JSON object: {response}"""
        #         json_response = query_llm(prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, return_json=True, json_schema=json_schema, logprobs=logprobs, system_prompt_included=False, is_hippa=is_hippa, debug=debug)
        #         print("Turning text into JSON by brute force...")
        #         return json_response
    return None

def query_gpt(prompt: str | dict, model: str = 'gpt-4o-mini', max_tokens: int = 4000, temperature: float = 0, top_p: float = 0, logprobs: bool = False, return_json: bool = False, json_schema = None, system_prompt_included: bool = False, is_hippa: bool = False, debug: bool = False):
    """OpenAI API wrapper; For HIPPA compliance, use client_safe e.g. model='openai-gpt-4o-high-quota-chat'"""
    temp_client = client_safe if is_hippa else client

    if system_prompt_included:
        # Format chat prompt with system and user messages
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    if 'o3' in model or 'o1' in model or 'o4' in model:
        api_params = {
            "model": model,
            "reasoning_effort": "high",
            "messages": messages,
        }
    else:
        api_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": 0
        }
    if logprobs:
        api_params["logprobs"] = logprobs
        api_params["top_logprobs"] = 3

    if return_json:
        if json_schema is None:
            api_params["response_format"] = {"type": "json_object"}
            completion = temp_client.chat.completions.create(**api_params)
            response = completion.choices[0].message.content.strip()
        else:
            api_params["response_format"] = json_schema
            completion = client.beta.chat.completions.parse(**api_params)
            response = completion.choices[0].message.parsed
    else: 
        completion = temp_client.chat.completions.create(**api_params)
        response = completion.choices[0].message.content.strip()
    if debug:
        print(f"Response: {response}")
    if logprobs:
        return response, completion.choices[0].logprobs
    else:
        return response