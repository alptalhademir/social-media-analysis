import os
import pandas as pd
import numpy as np
import logging
import subprocess
import sys

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
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from google.cloud import storage



class OpinionMatcher:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized similarity model: {model_name}")
    
    def match_opinions_to_topics(self, topics_df, opinions_df, threshold=0.45):
        logger.info("Matching opinions to topics using similarity model")
        
        # Get embeddings for all topics
        topic_texts = topics_df['text'].tolist()
        topic_embeddings = self.model.encode(topic_texts, convert_to_tensor=True)
        
        # For validation only (not used in the matching algorithm)
        topic_ids = topics_df['topic_id'].tolist()
        
        # Get embeddings for all opinions
        opinion_texts = opinions_df['text'].tolist()
        opinion_embeddings = self.model.encode(opinion_texts, convert_to_tensor=True)
        
        # Calculate similarity scores
        similarity_scores = util.cos_sim(opinion_embeddings, topic_embeddings)
        
        # Match each opinion to its most similar topic
        topic_groups = {}
        match_validation = []  # For validation only
        
        for i, opinion_row in enumerate(opinions_df.iterrows()):
            _, opinion = opinion_row
            # Find most similar topic based on embedding similarity
            best_match_idx = torch.argmax(similarity_scores[i]).item()
            best_match_score = similarity_scores[i][best_match_idx].item()
            
            # Only include if similarity is above threshold
            if best_match_score >= threshold:
                # Use an arbitrary topic key (position in list)
                matched_topic_idx = best_match_idx
                matched_topic_text = topic_texts[best_match_idx]
                
                # Create topic group if it doesn't exist yet
                if matched_topic_idx not in topic_groups:
                    topic_groups[matched_topic_idx] = {
                        'topic_text': matched_topic_text,
                        'opinions': []
                    }
                
                # Add opinion to matched group
                topic_groups[matched_topic_idx]['opinions'].append({
                    'text': opinion['text'],
                    'type': opinion['type']
                })
                
                # For validation only - store actual and predicted topic_id
                match_validation.append({
                    'actual_topic_id': opinion['topic_id'],
                    'predicted_topic_id': topic_ids[best_match_idx]
                })
        
        # Evaluate matching accuracy
        if match_validation:
            correct_matches = sum(1 for match in match_validation 
                                if match['actual_topic_id'] == match['predicted_topic_id'])
            accuracy = correct_matches / len(match_validation)
            logger.info(f"Topic matching accuracy: {accuracy:.4f} ({correct_matches}/{len(match_validation)})")
        
        logger.info(f"Created {len(topic_groups)} topic groups with matched opinions")
        return topic_groups

def prepare_training_data(topic_groups):
    texts = []
    labels = []
    
    # 4-class mapping
    label_mapping = {
        "Claim": 0, 
        "Evidence": 1, 
        "Counterclaim": 2, 
        "Rebuttal": 3
    }
    
    # For each topic group
    for topic_id, group_data in topic_groups.items():
        topic_text = group_data['topic_text']
        opinions = group_data['opinions']
        
        # For each opinion in the topic
        for i, opinion in enumerate(opinions):
            # Get other opinions from same topic as context
            context_opinions = [op['text'] for j, op in enumerate(opinions) if j != i]
            context_text = "\n".join([f"Opinion: {text}" for text in context_opinions[:3]])  # Limit to 3 context opinions
            
            # Format the input text with topic and context opinions
            input_text = f"Topic: {topic_text}\n\nContext opinions:\n{context_text}\n\nClassify this opinion: {opinion['text']}"
            
            # Add to training data
            texts.append(input_text)
            labels.append(label_mapping[opinion['type']])
    
    logger.info(f"Prepared {len(texts)} training examples")
    return texts, labels

#Load data and match opinions
def load_and_match_data(topics_csv, opinions_csv):
    # Load CSV files
    logger.info(f"Loading data from {opinions_csv} and {topics_csv}")
    topics_df = pd.read_csv(topics_csv)
    opinions_df = pd.read_csv(opinions_csv)
    
    # Match opinions to topics without using topic_id as a feature
    matcher = OpinionMatcher()
    topic_groups = matcher.match_opinions_to_topics(topics_df, opinions_df)
    
    # Create training data
    texts, labels = prepare_training_data(topic_groups)
    
    return texts, labels, topic_groups

#Compute metrics for multi-class classification
def compute_metrics(eval_pred):
    
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Calculate precision, recall, f1 for each class and average
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    # Class names for clearer reporting
    class_names = ["Claim", "Evidence", "Counterclaim", "Rebuttal"]
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(4))
    )
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[f"{class_name}_precision"] = precision_per_class[i]
        class_metrics[f"{class_name}_recall"] = recall_per_class[i]
        class_metrics[f"{class_name}_f1"] = f1_per_class[i]
        class_metrics[f"{class_name}_support"] = support_per_class[i]
    
    # Combine all metrics
    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        **class_metrics
    }
    
    return metrics

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

def main():
    try:
        # Get data paths
        bucket_name = os.getenv('BUCKET_NAME')
        if not bucket_name:
            raise ValueError("BUCKET_NAME environment variable not set")
        
        topics_csv = f"gs://{bucket_name}/data/topics.csv"
        opinions_csv = f"gs://{bucket_name}/data/opinions.csv"
        
        # Load and match data using semantic similarity
        texts, labels, topic_groups = load_and_match_data(topics_csv, opinions_csv)
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Initialize tokenizer and model for 4-class classification
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=4
        )
        
        # Create datasets
        def convert_to_hf_dataset(texts, labels, tokenizer):
            encodings = tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=256,
                return_tensors=None
            )
            encodings['labels'] = labels
            dataset = Dataset.from_dict(encodings)
            return dataset

        logger.info("Converting to HF Dataset format...")
        train_dataset = convert_to_hf_dataset(train_texts, train_labels, tokenizer)
        val_dataset = convert_to_hf_dataset(val_texts, val_labels, tokenizer)
        logger.info("Dataset conversion complete")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./working/checkpoints",
            num_train_epochs=4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./working/logs",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            fp16=True,
            logging_steps=100,
            dataloader_num_workers=1,
            dataloader_pin_memory=True
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Save classifier model
        gcs_uri = os.getenv('CLASSIFIER_ARTIFACT_URI')
        if not gcs_uri:
            raise ValueError("CLASSIFIER_ARTIFACT_URI environment variable not set")
            
        trainer.save_model("./working/final_model")
        tokenizer.save_pretrained("./working/final_model")
        upload_to_gcs("./working/final_model", gcs_uri)
        upload_to_gcs("./working/checkpoints", f"{gcs_uri}/checkpoints")
        
        # Save the sentence transformer model for inference
        sentence_model_dir = "./working/final_sentence_model"
        matcher = OpinionMatcher()
        matcher.model.save(sentence_model_dir)
        upload_to_gcs(sentence_model_dir, f"{gcs_uri}/sentence_model")
        
        logger.info("Training complete")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()