import os
import sys
import subprocess
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Remove TPU packages BEFORE any other imports to prevent transformers bug
try:
    logger.info("Removing TPU-related packages...")
    packages_to_remove = ["torch-xla", "cloud-tpu-client", "libtpu-nightly"]
    for package in packages_to_remove:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
        logger.info(f"Successfully removed {package}")
    
    # Set environment variables to disable XLA
    os.environ["USE_TORCH_XLA"] = "0"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
except Exception as e:
    logger.warning(f"Error during package removal: {str(e)}")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import pandas as pd
from huggingface_hub import login
from sklearn.model_selection import train_test_split
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from google.cloud import storage


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")
login(HF_TOKEN)

def upload_to_gcs(local_path, gcs_uri):
    gcs_path = gcs_uri.replace("gs://", "")
    bucket_name = gcs_path.split("/")[0]
    prefix = "/".join(gcs_path.split("/")[1:])

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, _, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_path)
            gcs_path = os.path.join(prefix, relative_path)
            
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_file_path)
    
    logger.info(f"Uploaded {local_path} to {gcs_uri}")



def load_data(bucket_name):
    # Load and prepare structured data for conclusion training
    try:
        if not bucket_name:
            raise ValueError("BUCKET_NAME environment variable not set")

        # Load all data files
        gcs_path_topics = f"gs://{bucket_name}/data/topics.csv"
        gcs_path_opinions = f"gs://{bucket_name}/data/opinions.csv"
        gcs_path_conclusions = f"gs://{bucket_name}/data/conclusions.csv"
        
        logger.info("Loading data from GCS...")
        topics_df = pd.read_csv(gcs_path_topics)
        opinions_df = pd.read_csv(gcs_path_opinions)
        conclusions_df = pd.read_csv(gcs_path_conclusions)
        
        examples = {}
        
        # Process topics
        for _, row in topics_df.iterrows():
            topic_id = row['topic_id']
            examples[topic_id] = {
                'topic': row['text'], 
                'opinions': [],
                'conclusion': None
            }
        
        # Process opinions - store as dicts with text and type
        for _, row in opinions_df.iterrows():
            topic_id = row['topic_id']
            if topic_id in examples:
                opinion_entry = {
                    'text': row['text'],
                    'type': row['type']
                }
                examples[topic_id]['opinions'].append(opinion_entry)
        
        # Process conclusions
        for _, row in conclusions_df.iterrows():
            topic_id = row['topic_id']
            if topic_id in examples:
                examples[topic_id]['conclusion'] = row['text']
        
        # Format as structured examples and filter incomplete ones
        formatted_examples = []
        for topic_id, data in examples.items():
            if data['topic'] and data['opinions'] and data['conclusion']:
                formatted_text = f"Topic:\n{data['topic']}\n\nOpinions:\n"
                
                # Sort opinions by type to have a more consistent order
                sorted_opinions = sorted(data['opinions'], key=lambda x: x['type'])
                
                for i, opinion in enumerate(sorted_opinions):
                    formatted_text += f"Opinion {i+1} ({opinion['type']})- {opinion['text']}\n"
                
                formatted_text += f"\nConclusion:\n{data['conclusion']}"
                formatted_examples.append(formatted_text)
        
        logger.info(f"Created {len(formatted_examples)} structured examples")
        
        # Split into train/validation
        train_texts, val_texts = train_test_split(formatted_examples, test_size=0.1, random_state=42)
        return train_texts, val_texts
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

def prepare_dataset_for_lm(texts, tokenizer):
    # Prepare dataset for llm with prompt-completion pairs
    try:
        formatted_samples = []
        
        for text in texts:
            parts = text.split("\nConclusion:\n")
            if len(parts) == 2:
                instruction = parts[0].strip()
                conclusion = parts[1].strip()
                
                # Chat template format
                messages = [
                    {"role": "user", "content": f"Generate a concluding statement for the following topic and its opinions:\n\n{instruction}"},
                    {"role": "assistant", "content": conclusion}
                ]
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted_samples.append({"text": formatted_text})
        
        dataset = Dataset.from_list(formatted_samples)
        
        def tokenize_function(examples):
            results = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=1024,
                return_tensors="pt"
            )
            results["labels"] = results["input_ids"].clone()
            return results
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset
    
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {str(e)}")
        raise

#Compute custom metrics for evaluation
def compute_metrics(eval_preds):
    
    logits, labels = eval_preds
    
    # Convert to PyTorch tensors if they are NumPy arrays
    logits = torch.tensor(logits) if isinstance(logits, np.ndarray) else logits
    labels = torch.tensor(labels) if isinstance(labels, np.ndarray) else labels
    
    # Shift logits and labels
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    
    # Flatten and create mask for padding tokens
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    mask = flat_labels != -100
    
    # Accuracy
    predictions = flat_logits.argmax(dim=-1)
    accuracy = (predictions[mask] == flat_labels[mask]).float().mean().item()
    
    # Loss and Perplexity
    loss = torch.nn.CrossEntropyLoss()(flat_logits, flat_labels)
    perplexity = torch.exp(loss).item()
    
    return {
        "accuracy": accuracy, 
        "perplexity": perplexity, 
        "loss": loss.item()
    }

def main():
    try:
        logger.info(f"- AIP_TENSORBOARD_LOG_DIR: {os.getenv('AIP_TENSORBOARD_LOG_DIR', 'NOT SET')}")
        logger.info("Initializing tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        if tokenizer.pad_token is None:
            logger.info("Setting padding token to EOS token")
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=False,
            device_map="auto",
            max_memory={0: "38GB"}
        )

        # Add LoRA adapters to the model
        logger.info("Applying LoRA adapters...")
        lora_config = LoraConfig(
            r=4,  # Rank
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )

        # Apply LoRA adapters to the model
        model = get_peft_model(model, lora_config)
        model.train()
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")

        logger.info("Loading and preparing data...")
        train_texts, val_texts = load_data(os.getenv('BUCKET_NAME'))
        
        train_dataset = prepare_dataset_for_lm(train_texts, tokenizer)
        val_dataset = prepare_dataset_for_lm(val_texts, tokenizer)

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        training_args = TrainingArguments(
            output_dir="./working/checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir="./working/logs",
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            learning_rate=2e-4,
            warmup_steps=100,
            weight_decay=0.01,
            load_best_model_at_end=True,
            report_to=["tensorboard"],
            gradient_accumulation_steps=32,
            dataloader_num_workers=1,
            fp16=True,
            logging_strategy="steps",
            logging_steps=10,
            save_total_limit=2,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        logger.info("Starting training...")
        trainer.train()

        gcs_uri = os.getenv('LLM_ARTIFACT_URI')
        if not gcs_uri:
            raise ValueError("LLM_ARTIFACT_URI environment variable not set")
        
        model.save_pretrained("./working/final_model")
        tokenizer.save_pretrained("./working/final_model")
        upload_to_gcs("./working/final_model", gcs_uri)
        upload_to_gcs("./working/checkpoints", f"{gcs_uri}/checkpoints")

        logger.info(f"Model and tokenizer saved to {gcs_uri}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()