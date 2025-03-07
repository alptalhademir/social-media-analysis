import yaml
import json
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisProducer:
    def __init__(self, config_path: str = "local/config/kafka_config.yaml"):
        self.config = self._load_config(config_path)
        self.producer = self._create_producer()
        self.input_topic = self.config["kafka"]["topics"]["input"]

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise

    def _create_producer(self) -> KafkaProducer:
        try:
            kafka_config = self.config["kafka"]
            producer_config = self.config.get("producer", {})
            
            return KafkaProducer(
                bootstrap_servers=kafka_config["bootstrap_servers"],
                value_serializer=lambda m: json.dumps(m).encode("utf-8"),
                acks=producer_config.get("acks", "all"),
                retries=producer_config.get("retries", 3),
                batch_size=producer_config.get("batch_size", 16384),
                compression_type=producer_config.get("compression_type", "gzip"),
                linger_ms=producer_config.get("linger_ms", 1)
            )
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {str(e)}")
            raise

    def produce_analysis_request(self, topic_text: str, opinions: List[str]) -> None:
        # Produce an analysis request message to Kafka topic

        message = {
            "topic": topic_text,
            "opinions": [{"text": opinion} for opinion in opinions],
            "timestamp": datetime.now().isoformat(),
            "request_id": f"req-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }

        try:
            future = self.producer.send(self.input_topic, message)
            # Wait for message to be delivered
            future.get(timeout=10)
            logger.info(f"Successfully produced analysis request for topic: {topic_text[:30]}...")
        except KafkaError as e:
            logger.error(f"Failed to produce message: {str(e)}")
            raise
        finally:
            self.producer.flush()

    def close(self):
        # Close the producer connection
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer connection closed")

if __name__ == "__main__":
    try:
        producer = AnalysisProducer()
        
        # Example
        topic = "Should face on Mars be considered a natural landform?"
        opinions = [
            "I think that the face is a natural landform because there is no life on Mars that we have discovered yet",
            "If life was on Mars, we would know by now. The reason why I think it is a natural landform because, nobody live on Mars in order to create the figure.",
            "People thought that the face was formed by aliens because they thought that there was life on Mars.",
            "Though some say that life on Mars does exist, I think that there is no life on Mars."
        ]
        
        producer.produce_analysis_request(topic, opinions)
        logger.info("Example message sent successfully")
        
    except Exception as e:
        logger.error(f"Error in producer example: {str(e)}")
    finally:
        producer.close()