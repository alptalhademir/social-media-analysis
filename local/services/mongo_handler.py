from pymongo import MongoClient
import yaml
from datetime import datetime
from bson import ObjectId
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_mongo_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["mongo"]

class MongoHandler:
    def __init__(self, config_file="local/config/mongo_config.yaml"):
        config = load_mongo_config(config_file)
        self.client = MongoClient(config["uri"])
        self.db = self.client[config["database"]]
        self.comments_collection = self.db[config["comments_collection"]]
        self.results_collection = self.db[config["results_collection"]]
        self.analyses_collection = self.db["analyses"]

    def insert_comment(self, comment_data):
        result = self.comments_collection.insert_one(comment_data)
        return result.inserted_id
    
    def update_comment(self, comment_id, update_data):
        self.comments_collection.update_one({"_id": comment_id}, {"$set": update_data})

    def get_comment(self, comment_id):
        return self.comments_collection.find_one({"_id": comment_id})
    
    def store_analysis_result(self, text_input, classification, conclusion):
        #Store the analysis result including classification and conclusion
        result = self.results_collection.insert_one({
            'input': text_input,
            'classification': classification,
            'conclusion': conclusion,
            'timestamp': datetime.now()
        })
        return result.inserted_id

    def get_analysis_result(self, result_id):
        # Get an analysis result by ID
        return self.results_collection.find_one({"_id": result_id})
    
    # Methods for the topic-opinions-conclusion structure
    def store_cloud_result(self, result_data):
        # Add timestamp for tracking
        result_data['timestamp'] = datetime.now()
        
        # Store the complete result exactly as received
        result = self.analyses_collection.insert_one(result_data)
        logger.info(f"Stored cloud analysis with ID: {result.inserted_id}")
        return result.inserted_id
    
    def get_cloud_result(self, result_id):
        # Get a cloud analysis result by ID
        if isinstance(result_id, str):
            result_id = ObjectId(result_id)
        return self.analyses_collection.find_one({"_id": result_id})
    
    def get_recent_analyses(self, limit=10):
        # Get the most recent analyses
        return list(self.analyses_collection.find().sort("timestamp", -1).limit(limit))
    
    def get_analyses_by_topic_substring(self, topic_substring):
        # Find analyses containing a substring in the topic
        return list(self.analyses_collection.find(
            {"topic": {"$regex": topic_substring, "$options": "i"}}
        ).sort("timestamp", -1))
    
    def store_user_input(self, topic, opinions):
        # Store the user input before sending to cloud
        input_data = {
            'topic': topic,
            'opinions': [{'opinion': op} for op in opinions],
            'timestamp': datetime.now(),
            'processed': False
        }
        result = self.analyses_collection.insert_one(input_data)
        logger.info(f"Stored user input with ID: {result.inserted_id}")
        return result.inserted_id
    
    def update_with_analysis_result(self, input_id, cloud_result):
        # Update an existing input record with the analysis results
        
        if isinstance(input_id, str):
            input_id = ObjectId(input_id)
            
        # Extract opinions with their types and conclusion
        opinions = cloud_result.get("opinions", [])
        conclusion = cloud_result.get("conclusion", "")
        
        # Update the document
        self.analyses_collection.update_one(
            {"_id": input_id},
            {
                "$set": {
                    "opinions": opinions,
                    "conclusion": conclusion,
                    "processed": True,
                    "processed_at": datetime.now()
                }
            }
        )
        logger.info(f"Updated input {input_id} with analysis results")
        return input_id