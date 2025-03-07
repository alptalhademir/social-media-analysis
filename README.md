# Social Media Analysis

A social media analysis solution for grouping comments by topic, classifying, and generating conclusions from user opinions


## Setup Requirements

- Google Cloud account with Vertex AI enabled
- MongoDB instance
- Apache Kafka
- Python 3.8+
- Required API keys and credentials in .env file
- key.json file for service account credentials
- Kafka and Mongo config files

## Setup

Clone the repository and change to that directory:

```
https://github.com/alptalhademir/social-media-analysis.git
cd social-media-analysis
```

Environment Configuration
```
# Google Cloud
PROJECT_ID=your-project-id
REGION=region
BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=./key.json

# Model endpoints
CLASSIFIER_ENDPOINT_URL=https://your-endpoint-url
LLM_ENDPOINT_URL=https://your-endpoint-url

# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE=digitalpulse

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

CLASSIFIER_ARTIFACT_URI=your-bucket-url
LLM_ARTIFACT_URI=your-bucket-url

SERVICE_ACCOUNT_EMAIL=service-account
AIP_TENSORBOARD_LOG_DIR=tensorboard-log-dir
TB_URI_LLM=tensorboard-instance-resource-name
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Train the models:
```bash
python deployment/vertex_ai/deploy_models.py
```