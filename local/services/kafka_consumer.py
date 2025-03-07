import yaml
import json
import logging
import grpc
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from collections import defaultdict
from datetime import datetime

import analysis_pb2
import analysis_pb2_grpc
from local.services.mongo_handler import MongoHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisConsumer:
    def __init__(self, config_path="local/config/kafka_config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize Kafka components
        self._init_kafka()
        
        # Initialize MongoDB handler
        self.mongo_handler = MongoHandler()
        
        # Initialize GRPC channel
        self.channel = grpc.insecure_channel(
            f"localhost:{self.config.get('grpc_port', 50051)}"
        )
        self.stub = analysis_pb2_grpc.AnalysisServiceStub(self.channel)
        logger.info("Analysis consumer initialized successfully")

    def _load_config(self, config_path):
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")
            raise

    def _init_kafka(self):
        kafka_config = self.config.get("kafka", {})
        consumer_config = self.config.get("consumer", {})
        
        input_topic = kafka_config.get("topics", {}).get("input", "social-media-comments")
        output_topic = kafka_config.get("topics", {}).get("output", "analysis-results")
        bootstrap_servers = kafka_config.get("bootstrap_servers", "localhost:9092")
        group_id = kafka_config.get("group_id", "task-consumer-group")
        
        # Initialize Kafka consumer
        try:
            self.consumer = KafkaConsumer(
                input_topic,
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset=consumer_config.get("auto_offset_reset", "earliest"),
                enable_auto_commit=consumer_config.get("enable_auto_commit", True),
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8"))
            )
            logger.info(f"Kafka consumer connected to topic: {input_topic}")
        except KafkaError as e:
            logger.error(f"Failed to initialize Kafka consumer: {str(e)}")
            raise
        
        # Initialize Kafka producer for results
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode("utf-8")
            )
            self.output_topic = output_topic
            logger.info(f"Kafka producer initialized for topic: {output_topic}")
        except KafkaError as e:
            logger.error(f"Failed to initialize Kafka producer: {str(e)}")
            raise

    def _process_message(self, message):
        # Process an incoming message from Kafka
        try:
            # Extract topic and opinions from the message
            topic_text = message.get("topic")
            opinions = message.get("opinions", [])
            request_id = message.get("request_id", f"req-{datetime.now().strftime('%Y%m%d%H%M%S')}")
            
            # Validate input
            if not topic_text or not opinions:
                logger.warning(f"Skipping invalid message: missing topic or opinions. Request ID: {request_id}")
                return
            
            logger.info(f"Processing request ID: {request_id}, topic: '{topic_text[:30]}...', opinions: {len(opinions)}")
            
            # Store initial request in MongoDB
            request_record_id = self.mongo_handler.store_user_input(topic_text, [op.get("text", "") for op in opinions])
            
            # Create gRPC request for analysis
            grpc_request = analysis_pb2.AnalysisRequest(
                topic=topic_text
            )
            
            # Add each opinion to the request
            for op in opinions:
                opinion_text = op.get("text", "")
                if opinion_text:
                    grpc_request.opinions.append(analysis_pb2.Opinion(text=opinion_text))
            
            # Call the gRPC service
            try:
                logger.info(f"Sending analysis request to gRPC service. Request ID: {request_id}")
                analysis_response = self.stub.AnalyzeOpinions(grpc_request)
                
                # Process the response
                if analysis_response:
                    # Convert gRPC response to dict for MongoDB
                    result = {
                        "topic": topic_text,
                        "opinions": [],
                        "conclusion": analysis_response.conclusion,
                        "formatted_text": analysis_response.formatted_text,
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Extract opinions with their types
                    for group in analysis_response.groups:
                        for opinion in group.opinions:
                            result["opinions"].append({
                                "opinion": opinion.text,
                                "type": opinion.type
                            })
                    
                    # Update MongoDB with analysis results
                    self.mongo_handler.update_with_analysis_result(request_record_id, result)
                    
                    # Send result to Kafka output topic
                    self.producer.send(self.output_topic, result)
                    self.producer.flush()
                    
                    logger.info(f"Analysis complete for request ID: {request_id}, conclusion length: {len(result['conclusion'])}")
                else:
                    logger.warning(f"No response from gRPC service for request ID: {request_id}")
            
            except grpc.RpcError as e:
                logger.error(f"gRPC service error for request ID {request_id}: {str(e)}")
                # Handle dead letter queue if needed
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

    def start_consuming(self):
        logger.info("Starting Kafka consumer...")
        try:
            for message in self.consumer:
                self._process_message(message.value)
        except KeyboardInterrupt:
            logger.info("Kafka consumer shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in consumer: {str(e)}")
        finally:
            self._cleanup()

    def _cleanup(self):
        # Clean up resources
        logger.info("Cleaning up resources...")
        if hasattr(self, 'consumer'):
            self.consumer.close()
        if hasattr(self, 'producer'):
            self.producer.close()
        if hasattr(self, 'channel'):
            self.channel.close()

if __name__ == "__main__":
    consumer = AnalysisConsumer()
    consumer.start_consuming()