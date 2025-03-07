import os
from dotenv import load_dotenv
from google.cloud import aiplatform, storage
from google.api_core import retry
import logging
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key.json'

# Configuration
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
CLASSIFIER_ARTIFACT_URI = os.getenv("CLASSIFIER_ARTIFACT_URI")
LLM_ARTIFACT_URI = os.getenv("LLM_ARTIFACT_URI")
PACKAGED_CLASSIFIER_URI = f"{CLASSIFIER_ARTIFACT_URI}_packaged"
PACKAGED_LLM_URI = f"{LLM_ARTIFACT_URI}_packaged"
TB_URI_LLM = os.getenv("TB_URI_LLM")

def package_models():
    # Package models with their handlers before deployment
    try:
        logger.info("Packaging classifier model with handler...")
        
        # Load the classifier packaging module
        spec = importlib.util.spec_from_file_location(
            "package_classifier", 
            "model/packaging/package_classifier.py"
        )
        classifier_packager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(classifier_packager)
        
        # Package the classifier model - update function to support new structure
        classifier_handler = "model/handlers/classifier_handler.py"
        classifier_packager.package_classifier_model_for_deployment(
            CLASSIFIER_ARTIFACT_URI,
            PACKAGED_CLASSIFIER_URI,
            classifier_handler,
            sentence_model_dir=f"{CLASSIFIER_ARTIFACT_URI}/sentence_model"
        )
        
        logger.info("Packaging LLM model with handler...")
        
        # Load the LLM packaging module
        spec = importlib.util.spec_from_file_location(
            "package_llm", 
            "model/packaging/package_llm.py"
        )
        llm_packager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llm_packager)
        
        # Package the LLM model - this is now a LoRA adapter
        llm_handler = "model/handlers/llm_handler.py"
        llm_packager.package_model_for_deployment(
            LLM_ARTIFACT_URI, 
            PACKAGED_LLM_URI, 
            llm_handler
        )
        
        logger.info("Models packaged successfully")
        
    except Exception as e:
        logger.error(f"Failed to package models: {str(e)}")
        raise

@retry.Retry()
def deploy_classifier():
    # Deploy the classifier model to Vertex AI
    try:
        aiplatform.init(
            project=PROJECT_ID, 
            location=REGION,
            staging_bucket=f"gs://{os.getenv('BUCKET_NAME')}"
        )
        classifier_model = aiplatform.Model.upload(
            display_name="task_classifier",
            artifact_uri=PACKAGED_CLASSIFIER_URI,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-4:latest",
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_environment_variables={
                "HANDLER_MODULE": "custom_handler",
                "PREDICT_FUNCTION": "predict",
            }
        )
        classifier_endpoint = classifier_model.deploy(
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            min_replica_count=1,
            max_replica_count=3,
        )
        logger.info(f"Classifier deployed to endpoint: {classifier_endpoint.resource_name}")
        return classifier_endpoint.resource_name
    except Exception as e:
        logger.error(f"Failed to deploy classifier: {str(e)}")
        raise

@retry.Retry()
def deploy_llm():
    # Deploy the LLM model to Vertex AI
    try:
        aiplatform.init(
            project=PROJECT_ID, 
            location=REGION,
            staging_bucket=f"gs://{os.getenv('BUCKET_NAME')}"
        )
        llm_model = aiplatform.Model.upload(
            display_name="task_llm",
            artifact_uri=PACKAGED_LLM_URI,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-4:latest",
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_environment_variables={
                "HANDLER_MODULE": "custom_handler",
                "PREDICT_FUNCTION": "predict",
            }
        )
        llm_endpoint = llm_model.deploy(
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            min_replica_count=1,
            max_replica_count=3,
        )
        logger.info(f"LLM deployed to endpoint: {llm_endpoint.resource_name}")
        return llm_endpoint.resource_name
    except Exception as e:
        logger.error(f"Failed to deploy LLM: {str(e)}")
        raise

def upload_data_to_gcs():
    #Upload data files to Google Cloud Storage"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(os.getenv('BUCKET_NAME'))
        
        # Data files to upload
        data_files = {
            'topics.csv': 'data/topics.csv',
            'opinions.csv': 'data/opinions.csv',
            'conclusions.csv': 'data/conclusions.csv'
        }
        
        for cloud_path, local_path in data_files.items():
            blob = bucket.blob(f'data/{cloud_path}')
            blob.upload_from_filename(local_path)
            logger.info(f'Uploaded {local_path} to gs://{bucket.name}/data/{cloud_path}')
            
    except Exception as e:
        logger.error(f"Failed to upload data: {str(e)}")
        raise

def submit_training_jobs():
    #Submit training jobs to Vertex AI
    try:
        aiplatform.init(
            project=PROJECT_ID, 
            location=REGION,
            staging_bucket=f"gs://{os.getenv('BUCKET_NAME')}"
        )

        # First upload training scripts to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(os.getenv('BUCKET_NAME'))

        # Upload training scripts
        training_scripts = [
            'model/training/fine_tune_classifier.py',
            'model/training/fine_tune_llm.py'
        ]
        
        for script_path in training_scripts:
            file_name = os.path.basename(script_path)
            script_blob = bucket.blob(f'scripts/{file_name}')
            script_blob.upload_from_filename(script_path)
            logger.info(f"Uploaded {script_path} to gs://{os.getenv('BUCKET_NAME')}/scripts/{file_name}")

        # Environment variables to pass to the training container
        training_env = {
            "PROJECT_ID": PROJECT_ID,
            "REGION": REGION,
            "BUCKET_NAME": os.getenv('BUCKET_NAME'),
            "HF_TOKEN": os.getenv('HF_TOKEN'),
            "CLASSIFIER_ARTIFACT_URI": CLASSIFIER_ARTIFACT_URI,
            "LLM_ARTIFACT_URI": LLM_ARTIFACT_URI,
            "AIP_TENSORBOARD_LOG_DIR": os.getenv('AIP_TENSORBOARD_LOG_DIR')
        }

        
        # Submit classifier training job
        classifier_job = aiplatform.CustomTrainingJob(
            display_name="classifier_training",
            script_path="model/training/fine_tune_classifier.py",
            container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest",
            requirements=["datasets", "transformers", "sentence_transformers", "accelerate"]
        )

        
        # Submit LLM training job
        llm_job = aiplatform.CustomTrainingJob(
            display_name="llm_training",
            script_path="model/training/fine_tune_llm.py",
            container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest",
            requirements=["datasets", "peft", "tensorboard"]
        )
        
        # Run classifier training
        classifier_run = classifier_job.run(
            args=["classifier"],
            environment_variables=training_env,
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            base_output_dir=CLASSIFIER_ARTIFACT_URI,
            sync=False
        )
        
        # Run LLM training
        llm_run = llm_job.run(
            args=[],
            environment_variables=training_env,
            machine_type="a2-highgpu-1g", 
            accelerator_type="NVIDIA_TESLA_A100",
            accelerator_count=1,
            base_output_dir=LLM_ARTIFACT_URI,
            sync=False,
            tensorboard=TB_URI_LLM,
            service_account=os.getenv("SERVICE_ACCOUNT_EMAIL")
        )

        return {
            "classifier_job": classifier_run,
            "llm_job": llm_run
        }
        
    except Exception as e:
        logger.error(f"Failed to submit training jobs: {str(e)}")
        raise

def validate_environment():
    #Validate all required environment variables are set
    required_vars = [
        "PROJECT_ID", "REGION", "BUCKET_NAME",
        "CLASSIFIER_ARTIFACT_URI", "LLM_ARTIFACT_URI"
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

if __name__ == "__main__":
    try:
        validate_environment()
        logger.info("Starting deployment process...")
        
        # Uupload data
        logger.info("Uploading data files...")
        upload_data_to_gcs()
        
        # Run training jobs
        logger.info("Submitting training jobs...")
        jobs = submit_training_jobs()

        logger.info("Waiting for training jobs to complete...")
        #jobs["classifier_job"].wait()
        #jobs["llm_job"].wait()
        """
        # Package models with handlers
        logger.info("Packaging models with handlers...")
        package_models()

        # Then deploy models
        logger.info("Deploying classifier model...")
        classifier_url = deploy_classifier()
        
        logger.info("Deploying LLM model...")
        llm_url = deploy_llm()
        
        logger.info("Deployment completed successfully")

        # Read existing .env content
        with open('.env', 'r') as f:
            env_content = f.read()
        
        # Update/add endpoint URLs
        env_content += f"\nCLASSIFIER_ENDPOINT_URL={classifier_url}"
        env_content += f"\nLLM_ENDPOINT_URL={llm_url}"

        # Write back to .env
        with open('.env', 'w') as f:
            f.write(env_content)
        
        logger.info("Endpoint URLs saved to .env file")"""
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        exit(1)