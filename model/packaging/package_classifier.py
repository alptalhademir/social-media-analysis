import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def package_classifier_model_for_deployment(source_dir, output_dir, handler_file, sentence_model_dir=None):
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectory for the main classifier model
        classifier_dir = os.path.join(output_dir, "classifier_model")
        os.makedirs(classifier_dir, exist_ok=True)
        
        # Copy main classifier model files
        logger.info(f"Copying classifier model files from {source_dir} to {classifier_dir}")
        for item in os.listdir(source_dir):
            # Skip the sentence_model directory if it exists in source_dir
            if item == "sentence_model":
                continue
                
            s = os.path.join(source_dir, item)
            d = os.path.join(classifier_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        
        # Handle sentence model if provided
        if sentence_model_dir:
            sentence_dir = os.path.join(output_dir, "sentence_model")
            os.makedirs(sentence_dir, exist_ok=True)
            
            logger.info(f"Copying sentence model files from {sentence_model_dir} to {sentence_dir}")
            for item in os.listdir(sentence_model_dir):
                s = os.path.join(sentence_model_dir, item)
                d = os.path.join(sentence_dir, item)
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
            
        # Install required packages for the classifier handler
        requirements_path = os.path.join(output_dir, "requirements.txt")
        with open(requirements_path, "w") as f:
            f.write("transformers>=4.30.0\n")
            f.write("torch>=2.3.1\n")
            f.write("sentence-transformers>=2.2.0\n")
            f.write("numpy>=2.0.0\n")
            f.write("scikit-learn>=1.2.0\n")
        
        logger.info(f"Classifier model packaged successfully at {output_dir}")
        return output_dir
    except Exception as e:
        logger.error(f"Error packaging classifier model: {str(e)}")
        raise

if __name__ == "__main__":
    # Get paths from environment variables or use defaults
    source_dir = os.getenv("CLASSIFIER_ARTIFACT_URI")
    output_dir = os.getenv("CLASSIFIER_ENDPOINT_URL")
    handler_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "handlers", "classifier_handler.py")
    
    # Get sentence model dir if available
    sentence_model_dir = os.getenv("SENTENCE_MODEL_URI", os.path.join(source_dir, "sentence_model"))
    
    # Package the classifier model
    package_classifier_model_for_deployment(source_dir, output_dir, handler_file, sentence_model_dir)