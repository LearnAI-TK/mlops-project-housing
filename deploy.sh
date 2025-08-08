#!/bin/bash

# deploy.sh - MLOps EC2 Deployment Script with MLflow + FastAPI + On-Demand Training

echo "🚀 Starting EC2 deployment..."
echo "📅 $(date)"
echo "🖥️  Host: $(hostname)"

# --- Configuration ---
DOCKERHUB_USERNAME="learnaitk"
DEPLOYMENT_NAME="mlops-$(date +%Y%m%d-%H%M%S)"

MLFLOW_IMAGE_NAME="ghcr.io/mlflow/mlflow:v2.18.0"
API_IMAGE_NAME="${DOCKERHUB_USERNAME}/mlops-project-api:latest"
TRAIN_IMAGE_NAME="${DOCKERHUB_USERNAME}/mlops-project-train:latest"

PERSISTENT_BASE_DIR="$HOME/mlops_persistent"
PERSISTENT_MLFLOW_DIR="${PERSISTENT_BASE_DIR}/mlflow"
PERSISTENT_API_LOGS_DIR="${PERSISTENT_BASE_DIR}/logs"
PERSISTENT_API_MODELS_DIR="${PERSISTENT_BASE_DIR}/models"

mkdir -p "$PERSISTENT_BASE_DIR"
LOG_FILE="${PERSISTENT_BASE_DIR}/deployment_${DEPLOYMENT_NAME}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "📝 Logging deployment to: $LOG_FILE"

# Get Docker host IP (for Linux)
DOCKER_HOST_IP=$(ip route | awk '/default/ {print $3}')

echo "Docker Host IP: $DOCKER_HOST_IP"

# --- System Check ---
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ Required command '$1' not found"
        exit 1
    fi
}

check_command curl
check_command docker

# --- Docker Installation ---
install_docker() {
    echo "🐳 Installing Docker..."
    if command -v apt &> /dev/null; then
        sudo apt update -y
        sudo apt install -y docker.io
    elif command -v yum &> /dev/null; then
        sudo yum update -y
        sudo yum install -y docker
    else
        echo "❌ Unsupported package manager"
        exit 1
    fi

    sudo systemctl start docker
    sudo systemctl enable docker
    echo "✅ Docker installed and started"
}

if ! systemctl is-active --quiet docker; then
    install_docker
else
    echo "✅ Docker is already running"
fi

# --- Docker Group ---
configure_docker_group() {
    if ! groups $USER | grep -q '\bdocker\b'; then
        echo "➕ Adding $USER to docker group..."
        sudo usermod -aG docker $USER
        newgrp docker || true
        echo "⚠️  You may need to restart your SSH session for group changes to take effect"
    else
        echo "✅ User already in docker group"
    fi
}
configure_docker_group

# --- Prepare Volumes ---
echo "📂 Setting up persistent directories..."
mkdir -p "$PERSISTENT_MLFLOW_DIR/mlruns" \
         "$PERSISTENT_API_LOGS_DIR" \
         "$PERSISTENT_API_MODELS_DIR"

touch "${PERSISTENT_MLFLOW_DIR}/mlflow.db"
chmod a+rw "${PERSISTENT_MLFLOW_DIR}/mlflow.db"

# --- Pull Images ---
pull_image() {
    local image=$1
    for i in {1..3}; do
        echo "⬇️  Pulling $image (attempt $i)..."
        if sudo docker pull $image; then
            echo "✅ Pulled $image"
            return 0
        fi
        sleep 10
    done
    echo "❌ Failed to pull $image"
    return 1
}

echo "📦 Pulling Docker images..."
pull_image "$MLFLOW_IMAGE_NAME" || exit 1
pull_image "$API_IMAGE_NAME" || exit 1
pull_image "$TRAIN_IMAGE_NAME" || exit 1

# --- Cleanup ---
cleanup_container() {
    local name=$1
    echo "🧹 Cleaning up $name..."
    if sudo docker ps -a | grep -q $name; then
        sudo docker stop $name || true
        sudo docker rm $name || true
    fi
}
cleanup_container "mlflow-server"
cleanup_container "housing-api"
cleanup_container "training-job"

# --- Deploy MLflow ---
echo "🚀 Starting MLflow..."
sudo docker run -d \
  --name mlflow-server \
  --restart unless-stopped \
  -p 5000:5000 \
  -v "${PERSISTENT_MLFLOW_DIR}:/mlflow" \
  -e MLFLOW_BACKEND_STORE_URI="sqlite:////mlflow/mlflow.db" \
  -e MLFLOW_DEFAULT_ARTIFACT_ROOT="/mlflow/mlruns" \
  $MLFLOW_IMAGE_NAME \
  bash -c "mkdir -p /mlflow/mlruns && touch /mlflow/mlflow.db && chmod a+rw /mlflow/mlflow.db && mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/mlruns"

sleep 25

# --- Get EC2 IP ---
EC2_HOST_IP=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/local-ipv4)
if [ -z "$EC2_HOST_IP" ]; then
    EC2_HOST_IP=$(ip route get 1 | awk '{print $7; exit}')
    echo "⚠️  Could not fetch EC2 metadata, fallback IP: $EC2_HOST_IP"
else
    echo "🌐 EC2 private IP: $EC2_HOST_IP"
fi
MLFLOW_URI_FOR_API="http://${EC2_HOST_IP}:5000"

# --- Deploy API ---
echo "🚀 Starting FastAPI container..."
sudo docker run -d \
  --name housing-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -v "${PERSISTENT_API_LOGS_DIR}:/app/logs" \
  -v "${PERSISTENT_API_MODELS_DIR}:/app/models" \
  -v "${PERSISTENT_MLFLOW_DIR}:/mlflow" \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  -e MODEL_NAME="CaliforniaHousingRegressor" \
  -e MODEL_ALIAS="staging" \
  -e PYTHONUNBUFFERED="1" \
  $API_IMAGE_NAME

# --- Health Checks ---
verify_container() {
    local name=$1
    local port=$2
    for i in {1..12}; do
        if curl -sSf "http://localhost:$port/" >/dev/null 2>&1; then
            echo "✅ $name is running on port $port"
            return 0
        fi
        echo "⏳ $name not ready (attempt $i)..."
        sleep 5
    done
    echo "❌ $name failed to start"
    sudo docker logs $name --tail 50
    return 1
}

verify_container "mlflow-server" 5000 || exit 1
verify_container "housing-api" 8000 || exit 1

# --- Done ---
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo "<your-public-ip>")

echo "🎉 Deployment successful!"
echo "🌍 MLflow UI:     http://$PUBLIC_IP:5000"
echo "🚀 API Docs:      http://$PUBLIC_IP:8000/docs"
echo "📊 Metrics:       http://$PUBLIC_IP:8000/metrics"
echo "📁 Logs stored in: $PERSISTENT_BASE_DIR"

echo "🏋️  To run training:"
echo "sudo docker run --rm --name training-job \\"
echo "  -v ${PERSISTENT_API_LOGS_DIR}:/app/logs \\"
echo "  -v ${PERSISTENT_API_MODELS_DIR}:/app/models \\"
echo "  -v ${PERSISTENT_MLFLOW_DIR}:/mlflow \\"
echo "  -e MLFLOW_TRACKING_URI=${MLFLOW_URI_FOR_API} \\"
echo "  -e MODEL_NAME=CaliforniaHousingRegressor \\"
echo "  -e MODEL_ALIAS=staging \\"
echo "  $TRAIN_IMAGE_NAME \\"
echo "  python src/model_training.py"
