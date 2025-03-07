import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and tokenizer
model = None
tokenizer = None
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def load_model():
    global model, tokenizer
    try:
        logger.info("Loading LLM model and tokenizer...")
        
        model_path = "/models"
        
        # Check if this is a LoRA adapter
        is_lora_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_lora_adapter:
            logger.info("Loading as LoRA adapter...")
            # Load base model and tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            
            # Set appropriate device and precision
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # Load base model with reduced precision
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Load the LoRA adapter on top of the base model
            model = PeftModel.from_pretrained(
                base_model,
                model_path
            )
        else:
            # Regular model loading (non-LoRA)
            logger.info("Loading as full model...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Set appropriate device and precision
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # Load model with inference optimizations
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        model.eval()
        logger.info(f"Model and tokenizer loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def predict(instances):
    
    global model, tokenizer
    
    # Load model if not already loaded
    if model is None or tokenizer is None:
        success = load_model()
        if not success:
            return ["Failed to load model"] * len(instances)
    
    results = []
    for instance in instances:
        try:
            # Get the input text
            text = instance.get("text", "")
            
            # Use chat template
            messages = [
                {"role": "user", "content": f"Generate a concluding statement for the following topic and its opinions:\n\n{text}"}
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            }
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, **gen_kwargs)
                
            # Decode and extract conclusion using the chat template logic
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract the assistant's response
            if "assistant" in generated_text.lower():
                # Try to extract the assistant's part if using a chat template with role markers
                conclusion = generated_text.split("assistant")[-1].strip()
                # Remove any leading punctuation like ":"
                conclusion = conclusion.lstrip(" :\"'")
            else:
                # If no clear marker, get everything after the user's prompt
                user_prompt = f"Generate a concluding statement for the following topic and its opinions:\n\n{text}"
                if user_prompt in generated_text:
                    conclusion = generated_text.split(user_prompt)[-1].strip()
                else:
                    conclusion = generated_text
            
            results.append(conclusion)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            results.append(f"Error generating conclusion: {str(e)}")
    
    return results