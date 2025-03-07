import os
import grpc
import yaml
import logging
import json
import requests
from concurrent import futures
import time
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from local.services.mongo_handler import MongoHandler

import analysis_pb2
import analysis_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def load_config(config_file="local/config/kafka_config.yaml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

class AnalysisService(analysis_pb2_grpc.AnalysisServiceServicer):
    def __init__(self):
        # Initialize Vertex AI client
        self.project = os.environ.get("PROJECT_ID")
        self.location = os.environ.get("REGION")
        
        # Get endpoint URLs from environment variables
        self.classifier_endpoint_url = os.environ.get("CLASSIFIER_ENDPOINT_URL")
        self.llm_endpoint_url = os.environ.get("LLM_ENDPOINT_URL")
        
        if not self.classifier_endpoint_url or not self.llm_endpoint_url:
            logger.error("Missing endpoint URLs in environment variables")
            raise ValueError("CLASSIFIER_ENDPOINT_URL and LLM_ENDPOINT_URL must be set")
        
        # Initialize MongoDB handler
        self.mongo = MongoHandler()
        
        # Get credentials for API requests
        try:
            # If using a service account key file
            if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                self.credentials = service_account.Credentials.from_service_account_file(
                    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            else:
                # Use default credentials
                import google.auth
                self.credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            
            # Make sure credentials are fresh
            if self.credentials.expired:
                self.credentials.refresh(Request())
                
            logger.info("Successfully initialized credentials for Vertex AI endpoints")
        except Exception as e:
            logger.error(f"Failed to initialize credentials: {e}")
            raise
    
    def _get_auth_headers(self):
        # Get authentication headers for API requests
        # Refresh token if needed
        if self.credentials.expired:
            self.credentials.refresh(Request())
            
        return {
            "Authorization": f"Bearer {self.credentials.token}",
            "Content-Type": "application/json"
        }
    
    def AnalyzeOpinions(self, request, context):
        # Workflow: grouping, classification, and conclusion.
        try:
            topic = request.topic
            opinions = [{"text": op.text} for op in request.opinions]
            
            logger.info(f"Processing analysis request with topic: '{topic[:30]}...' and {len(opinions)} opinions")
            
            # Step 1: Call classifier endpoint for grouping and classification
            classifier_input = {
                "instances": [{
                    "topic": topic,
                    "opinions": opinions
                }]
            }
            
            # Call classifier endpoint via REST API
            headers = self._get_auth_headers()
            logger.info(f"Calling classifier endpoint: {self.classifier_endpoint_url}")
            
            classifier_response = requests.post(
                self.classifier_endpoint_url,
                headers=headers,
                data=json.dumps(classifier_input)
            )
            
            if classifier_response.status_code != 200:
                error_msg = f"Classifier endpoint returned error: {classifier_response.status_code}, {classifier_response.text}"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
                return analysis_pb2.AnalysisResponse()
            
            classifier_result = classifier_response.json()
            
            # Extract formatted text for LLM
            if classifier_result and "predictions" in classifier_result and classifier_result["predictions"]:
                formatted_text = classifier_result["predictions"][0].get("formatted_text")
                grouped_opinions = classifier_result["predictions"][0].get("groups", [])
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("No valid response from classifier endpoint")
                return analysis_pb2.AnalysisResponse()
            
            # Step 2: Call LLM endpoint for conclusion generation
            llm_input = {
                "instances": [{"text": formatted_text}]
            }
            
            logger.info(f"Calling LLM endpoint: {self.llm_endpoint_url}")
            llm_response = requests.post(
                self.llm_endpoint_url,
                headers=headers,
                data=json.dumps(llm_input)
            )
            
            if llm_response.status_code != 200:
                error_msg = f"LLM endpoint returned error: {llm_response.status_code}, {llm_response.text}"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
                return analysis_pb2.AnalysisResponse()
            
            llm_result = llm_response.json()
            
            conclusion = llm_result.get("predictions", [""])[0]
            
            # Step 3: Store the complete result in MongoDB
            result_data = {
                "topic": topic,
                "opinions": [],
                "conclusion": conclusion
            }
            
            # Convert grouped opinions to the required format
            for group in grouped_opinions:
                for opinion in group.get("opinions", []):
                    result_data["opinions"].append({
                        "opinion": opinion.get("text", ""),
                        "type": opinion.get("type", "Unknown")
                    })
            
            # Store in MongoDB
            self.mongo.store_cloud_result(result_data)
            
            # Step 4: Build and return the response
            response = analysis_pb2.AnalysisResponse(
                topic=topic,
                conclusion=conclusion,
                formatted_text=formatted_text
            )
            
            # Add opinion groups to response
            for group in grouped_opinions:
                opinion_group = analysis_pb2.OpinionGroup(
                    group_id=group.get("group_id", "")
                )
                
                for op in group.get("opinions", []):
                    opinion_group.opinions.append(analysis_pb2.Opinion(
                        text=op.get("text", ""),
                        type=op.get("type", "")
                    ))
                
                response.groups.append(opinion_group)
            
            logger.info(f"Successfully processed analysis request, generated conclusion of length {len(conclusion)}")
            return response
            
        except Exception as e:
            logger.error(f"Error in AnalyzeOpinions: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Analysis failed: {str(e)}")
            return analysis_pb2.AnalysisResponse()
    
    def ClassifyOpinion(self, request, context):
        # Classify a single opinion - simpler endpoint for backward compatibility
        try:
            topic = request.topic
            opinion_text = request.opinion_text
            
            logger.info(f"Processing classification request for opinion: '{opinion_text[:30]}...'")
            
            # Call classifier endpoint for single opinion
            classifier_input = {
                "instances": [{
                    "topic": topic,
                    "text": opinion_text
                }]
            }
            
            # Call classifier endpoint via REST API
            headers = self._get_auth_headers()
            
            classifier_response = requests.post(
                self.classifier_endpoint_url,
                headers=headers,
                data=json.dumps(classifier_input)
            )
            
            if classifier_response.status_code != 200:
                error_msg = f"Classifier endpoint returned error: {classifier_response.status_code}, {classifier_response.text}"
                logger.error(error_msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
                return analysis_pb2.ClassifyOpinionResponse()
            
            classifier_result = classifier_response.json()
            
            # Extract classification
            if classifier_result and "predictions" in classifier_result and classifier_result["predictions"]:
                opinion_type = classifier_result["predictions"][0].get("type", "Unknown")
                confidence = classifier_result["predictions"][0].get("confidence", 0.0)
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("No valid response from classifier endpoint")
                return analysis_pb2.ClassifyOpinionResponse()
            
            return analysis_pb2.ClassifyOpinionResponse(
                type=opinion_type,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error in ClassifyOpinion: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Classification failed: {str(e)}")
            return analysis_pb2.ClassifyOpinionResponse()

def serve():
    port = os.environ.get("GRPC_SERVER_PORT", "50051")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    analysis_pb2_grpc.add_AnalysisServiceServicer_to_server(AnalysisService(), server)
    
    # Bind only to localhost for increased security
    server.add_insecure_port(f"127.0.0.1:{port}")
    
    server.start()
    logger.info(f"GRPC Server is running on localhost:{port}...")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
        logger.info("Server stopped gracefully")

if __name__ == "__main__":
    serve()