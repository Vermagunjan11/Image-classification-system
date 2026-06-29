# VisionAI Deployment Guide

Complete guide to deploy VisionAI to production environments.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Streamlit Cloud](#streamlit-cloud)
4. [Hugging Face Spaces](#hugging-face-spaces)
5. [AWS Deployment](#aws-deployment)
6. [Performance Optimization](#performance-optimization)

---

## Local Deployment

### 1. Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/VisionAI.git
cd VisionAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run streamlit_app.py
```

### 2. Production Environment

```bash
# Install with production settings
pip install -r requirements.txt

# Set environment variables
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

# Run with gunicorn (optional)
pip install gunicorn
gunicorn -w 1 streamlit_app:app
```

---

## Docker Deployment

### 1. Build Docker Image

```bash
# Build image
docker build -t visionai:latest .

# Build with specific tag
docker build -t visionai:v1.0 .
```

### 2. Run Docker Container

```bash
# Basic run
docker run -p 8501:8501 visionai:latest

# With volume mounting
docker run -p 8501:8501 \
  -v $(pwd)/saved_models:/app/saved_models \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  visionai:latest

# With GPU support
docker run --gpus all -p 8501:8501 visionai:latest

# With environment variables
docker run -p 8501:8501 \
  -e PYTHONUNBUFFERED=1 \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  visionai:latest
```

### 3. Docker Compose Deployment

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f visionai

# Rebuild image
docker-compose up --build
```

### 4. Docker Registry Push

```bash
# Tag image for registry
docker tag visionai:latest username/visionai:latest

# Login to Docker Hub
docker login

# Push image
docker push username/visionai:latest
```

---

## Streamlit Cloud

### 1. Prepare for Deployment

```bash
# Create .streamlit/config.toml
mkdir -p .streamlit
cat > .streamlit/config.toml << EOF
[theme]
primaryColor = "#0066cc"
backgroundColor = "#f8f9fa"
secondaryBackgroundColor = "#e3f2fd"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableXsrfProtection = true
maxUploadSize = 200

[logger]
level = "info"
EOF
```

### 2. Push to GitHub

```bash
git add .
git commit -m "Add VisionAI deployment"
git push origin main
```

### 3. Deploy on Streamlit Cloud

1. Visit https://share.streamlit.io/
2. Click "New app"
3. Select repository, branch, and main file (`streamlit_app.py`)
4. Click "Deploy"

### 4. Share URL

Once deployed, your app will be available at:
```
https://share.streamlit.io/yourusername/VisionAI/streamlit_app.py
```

---

## Hugging Face Spaces

### 1. Create Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in details:
   - Owner: Your username
   - Space name: `vision-ai`
   - License: MIT
   - Space SDK: Docker
4. Click "Create Space"

### 2. Push Code

```bash
# Install git-lfs
git lfs install

# Clone space
git clone https://huggingface.co/spaces/yourusername/vision-ai
cd vision-ai

# Copy project files
cp -r ../VisionAI/* .

# Create Dockerfile (provided)
cp ../VisionAI/Dockerfile .

# Push to Hugging Face
git add .
git commit -m "Add VisionAI"
git push
```

### 3. Access Space

Your app will be available at:
```
https://huggingface.co/spaces/yourusername/vision-ai
```

---

## AWS Deployment

### 1. Using AWS EC2

```bash
# Launch EC2 instance
# - AMI: Ubuntu 22.04
# - Instance: t3.medium or larger
# - Storage: 30GB

# SSH into instance
ssh -i key.pem ec2-user@your-instance-ip

# Install Docker
sudo apt-get update
sudo apt-get install docker.io -y
sudo systemctl start docker

# Clone repository
git clone https://github.com/yourusername/VisionAI.git
cd VisionAI

# Build and run
docker build -t visionai .
docker run -d -p 80:8501 visionai
```

### 2. Using AWS AppRunner

1. Push Docker image to ECR:
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

docker tag visionai:latest \
  123456789.dkr.ecr.us-east-1.amazonaws.com/visionai:latest

docker push \
  123456789.dkr.ecr.us-east-1.amazonaws.com/visionai:latest
```

2. Create AppRunner service:
   - Source: ECR image
   - Port: 8501
   - Automatic deployment: On

### 3. Using AWS ECS

```bash
# Create task definition
aws ecs register-task-definition \
  --family visionai \
  --network-mode awsvpc \
  --requires-compatibilities FARGATE \
  --cpu 1024 \
  --memory 2048 \
  --container-definitions ...

# Create service
aws ecs create-service \
  --cluster default \
  --service-name visionai \
  --task-definition visionai
```

---

## Google Cloud Platform

### 1. Using Cloud Run

```bash
# Authenticate
gcloud auth login

# Configure project
gcloud config set project PROJECT_ID

# Build and deploy
gcloud run deploy visionai \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1
```

### 2. Using Compute Engine

```bash
# Create instance
gcloud compute instances create visionai-instance \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --machine-type=n1-standard-2 \
  --zone=us-central1-a

# SSH into instance
gcloud compute ssh visionai-instance

# Install and run (same as EC2)
```

---

## Microsoft Azure

### 1. Using Azure Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name visionai \
  --image visionai:latest \
  --ports 8501 \
  --memory 2
```

### 2. Using Azure App Service

```bash
# Create App Service Plan
az appservice plan create \
  --name visionai-plan \
  --resource-group myResourceGroup \
  --sku B2 \
  --is-linux

# Create Web App
az webapp create \
  --resource-group myResourceGroup \
  --plan visionai-plan \
  --name visionai-app \
  --deployment-container-image-name-user-name visionai
```

---

## Performance Optimization

### 1. Model Optimization

```python
# Use quantization
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### 2. Caching Strategies

```python
# Enable model caching
@st.cache_resource
def load_model(model_name):
    return tf.keras.models.load_model(f'saved_models/{model_name}.h5')

# Cache predictions
@st.cache_data
def predict_image(image_path, model_name):
    # Prediction logic
    pass
```

### 3. Load Balancing

```bash
# Using Nginx for load balancing
upstream visionai {
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}

server {
    listen 80;
    location / {
        proxy_pass http://visionai;
    }
}
```

### 4. Database for Predictions

```python
# Store predictions in database
import sqlite3

conn = sqlite3.connect('predictions.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE predictions (
        id INTEGER PRIMARY KEY,
        timestamp DATETIME,
        image_path TEXT,
        prediction TEXT,
        confidence REAL
    )
''')

# Insert prediction
cursor.execute(
    'INSERT INTO predictions VALUES (?, ?, ?, ?, ?)',
    (None, datetime.now(), path, class_name, confidence)
)

conn.commit()
```

---

## Monitoring and Logging

### 1. Logging

```bash
# View container logs
docker logs visionai

# Stream logs
docker logs -f visionai

# Save logs
docker logs visionai > logs.txt
```

### 2. Health Checks

```python
import healthcheck

@app.route('/health')
def health():
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }
```

### 3. Monitoring

```bash
# Monitor resource usage
docker stats visionai

# Monitor with Prometheus
# (Setup Prometheus + Grafana dashboard)
```

---

## Security Best Practices

### 1. Environment Variables

```bash
# Store secrets in .env
DB_PASSWORD=xxxxx
API_KEY=xxxxx

# Load in application
from dotenv import load_dotenv
import os

load_dotenv()
password = os.getenv('DB_PASSWORD')
```

### 2. HTTPS/SSL

```bash
# Generate certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Run with HTTPS
streamlit run streamlit_app.py \
  --logger.level=info \
  --server.sslCertFile=cert.pem \
  --server.sslKeyFile=key.pem
```

### 3. Authentication

```python
import streamlit as st

# Simple password authentication
def check_password():
    if st.session_state.get('password_correct'):
        return True
    
    password = st.text_input('Password:', type='password')
    if password == st.secrets['password']:
        st.session_state['password_correct'] = True
        return True
    return False

if not check_password():
    st.stop()
```

---

## Troubleshooting

### Issue: Port Already in Use

```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 PID
```

### Issue: Out of Memory

```bash
# Increase Docker memory limit
docker run -m 4g visionai

# Or modify docker-compose.yml
services:
  visionai:
    mem_limit: 4g
```

### Issue: Slow Inference

```bash
# Profile code
python -m cProfile -s cumulative train.py

# Use GPU
CUDA_VISIBLE_DEVICES=0 python train.py
```

---

## Summary

| Platform | Pros | Cons |
|----------|------|------|
| **Local** | Full control | Limited scalability |
| **Docker** | Portable | Requires Docker knowledge |
| **Streamlit Cloud** | Free tier | Limited resources |
| **Hugging Face** | Easy setup | Limited customization |
| **AWS** | Scalable | More expensive |
| **GCP** | Good free tier | Learning curve |
| **Azure** | Enterprise ready | Complex setup |

---

## Next Steps

1. **Choose deployment platform**
2. **Prepare environment**
3. **Test deployment**
4. **Setup monitoring**
5. **Configure backups**
6. **Document setup**

For more help, check the README.md and QUICKSTART.md files.
