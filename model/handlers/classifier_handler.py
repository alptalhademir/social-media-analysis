import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global models
classifier_model = None
classifier_tokenizer = None
similarity_model = None

LABEL_MAPPING = {0: "Claim", 1: "Evidence", 2: "Counterclaim", 3: "Rebuttal"}

def load_models():
    global classifier_model, classifier_tokenizer, similarity_model
    
    try:
        logger.info("Loading models...")
        model_dir = "/models"  # Vertex AI mounts models here
        
        # Load classifier from classifier_model subdirectory
        classifier_path = os.path.join(model_dir, "classifier_model")
        if not os.path.exists(classifier_path):
            # Fall back to original directory for backward compatibility
            classifier_path = model_dir
            
        logger.info(f"Loading classifier from {classifier_path}")
        classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
        classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_path)
        classifier_model.eval()
        
        # Load similarity model from sentence_model subdirectory or fall back to default
        similarity_path = os.path.join(model_dir, "sentence_model")
        if os.path.exists(similarity_path):
            logger.info(f"Loading sentence transformer from {similarity_path}")
            similarity_model = SentenceTransformer(similarity_path)
        else:
            logger.info("Using default sentence transformer model")
            similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return False

def group_opinions_by_similarity(opinions, similarity_threshold=0.7):
    
    if len(opinions) <= 1:
        return [opinions]
    
    # Extract text
    texts = [op["text"] for op in opinions]
    
    # Encode all opinions
    embeddings = similarity_model.encode(texts)
    
    # Group by similarity
    grouped_indices = []
    used = set()
    
    for i in range(len(texts)):
        if i in used:
            continue
            
        # Start a new group
        group = [i]
        used.add(i)
        
        # Find similar opinions
        for j in range(len(texts)):
            if j != i and j not in used:
                # Calculate similarity
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                if sim >= similarity_threshold:
                    group.append(j)
                    used.add(j)
                    
        if group:
            grouped_indices.append(group)
    
    # Convert indices to actual opinions
    grouped_opinions = []
    for group in grouped_indices:
        grouped_opinions.append([opinions[i] for i in group])
    
    return grouped_opinions

def classify_opinion(text, topic="", context=""):
    
    global classifier_model, classifier_tokenizer
    
    try:
        # Format input text to match the training format
        input_parts = []
        
        if topic:
            input_parts.append(f"Topic: {topic}")
        
        if context:
            input_parts.append(f"Context opinions:\n{context}")
            
        input_parts.append(f"Classify this opinion: {text}")
        
        input_text = "\n\n".join(input_parts)
        
        # Tokenize
        inputs = classifier_tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        # Run inference
        with torch.no_grad():
            outputs = classifier_model(**inputs)
        
        # Get prediction
        logits = outputs.logits.detach().cpu().numpy()
        pred_idx = np.argmax(logits, axis=1)[0]
        
        return {
            "type": LABEL_MAPPING[pred_idx],
            "confidence": float(logits[0][pred_idx])
        }
    except Exception as e:
        logger.error(f"Error in classify_opinion: {str(e)}")
        return {"type": "Unknown", "confidence": 0.0}

def format_output(grouped_opinions, topic_text):
    results = []
    
    for group_idx, group in enumerate(grouped_opinions, 1):
        if not group:
            continue
        
        # Format exactly as shown in the example
        formatted_output = f"Topic: {topic_text}\nOpinions:\n"
        
        # Add each classified opinion with the exact spacing
        for i, opinion in enumerate(group, 1):
            opinion_type = opinion.get("type", "Unknown")
            formatted_output += f"Opinion {i} ({opinion_type})- {opinion['text']} \n"
        
        # Remove trailing newline
        formatted_output = formatted_output.rstrip()
        
        results.append({
            "group_id": f"group_{group_idx}",
            "formatted_text": formatted_output,
            "opinions": group
        })
    
    return results

def predict(instances):
    global classifier_model, similarity_model
    
    if classifier_model is None or similarity_model is None:
        success = load_models()
        if not success:
            return [{"error": "Failed to load models"}]
    
    results = []
    
    for instance in instances:
        try:
            # Get topic from input
            topic_text = instance.get("topic", "")
            
            # Get input data
            if "opinions" in instance:
                # Get list of opinions
                opinions = instance["opinions"]
                
                # Add text field if not present (handle different input formats)
                for op in opinions:
                    if "text" not in op:
                        op["text"] = op.get("opinion", "")
                
                # Step 1: Group opinions by similarity
                grouped_opinions = group_opinions_by_similarity(opinions)
                
                # Step 2: Classify each opinion in each group
                for group in grouped_opinions:
                    # Format context from other opinions in the group
                    context_opinions = []
                    for op in group:
                        context_opinions.append(f"Opinion: {op['text']}")
                    context_text = "\n".join(context_opinions)
                    
                    # Classify each opinion with context
                    for opinion in group:
                        # Remove current opinion from context
                        current_context = "\n".join([line for line in context_opinions 
                                                  if opinion['text'] not in line])
                        classification = classify_opinion(
                            opinion["text"], 
                            topic=topic_text,
                            context=current_context
                        )
                        opinion.update(classification)
                
                # Step 3: Format output in requested structure
                formatted_results = format_output(grouped_opinions, topic_text)
                
                # Return exactly formatted text for primary response
                if formatted_results:
                    results.append({
                        "formatted_text": formatted_results[0]["formatted_text"],
                        "groups": formatted_results
                    })
                else:
                    results.append({
                        "formatted_text": f"Topic: {topic_text}\nOpinions:\n(No opinions classified)",
                        "groups": []
                    })
                
            else:
                # Single text to classify
                text = instance.get("text", "")
                classification = classify_opinion(
                    text,
                    topic=topic_text
                )
                
                # Format single opinion result
                formatted_text = (f"Topic: {topic_text}\nOpinions:\n" +
                                 f"Opinion 1 ({classification['type']})- {text}")
                
                results.append({
                    "formatted_text": formatted_text,
                    "classification": classification
                })
                
        except Exception as e:
            logger.error(f"Error processing instance: {str(e)}")
            results.append({"error": str(e)})
    
    return results