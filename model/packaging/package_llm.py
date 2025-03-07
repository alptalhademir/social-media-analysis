import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def package_model_for_deployment(source_dir, output_dir, handler_file):

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy all model files
        logger.info(f"Copying model files from {source_dir} to {output_dir}")
        for item in os.listdir(source_dir):
            s = os.path.join(source_dir, item)
            d = os.path.join(output_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        
        # Copy handler script as custom_handler.py
        logger.info(f"Copying handler from {handler_file} to {output_dir}/custom_handler.py")
        shutil.copy2(handler_file, os.path.join(output_dir, "custom_handler.py"))
        
        # Create config file for handler
        with open(os.path.join(output_dir, "config.py"), "w") as f:
            f.write("handler_file = 'custom_handler.py'\n")
            f.write("prediction_fn = 'predict'\n")

        with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
            f.write("transformers>=4.30.0\n")
            f.write("torch>=2.3.1\n")
            f.write("peft>=0.9.0\n")
        
        logger.info(f"Model packaged successfully at {output_dir}")
        return output_dir
    except Exception as e:
        logger.error(f"Error packaging model: {str(e)}")
        raise

if __name__ == "__main__":
    # Get paths from environment variables or use defaults
    source_dir = os.getenv("LLM_ARTIFACT_URI")
    output_dir = os.getenv("LLM_ENDPOINT_URL")
    handler_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "handlers", "llm_handler.py")
    
    package_model_for_deployment(source_dir, output_dir, handler_file)