# 🏠 MLOps Housing Price Prediction

## 📌 Overview
This project implements an **end-to-end MLOps pipeline** for predicting California housing prices.
It leverages **MLflow**, **Docker**, **FastAPI**, and **CI/CD with GitHub Actions**, and follows best practices for **model training, versioning, deployment, and monitoring**.

---

## 🚀 Features
- **Model Training**: Trains regression models on California Housing dataset.
- **Experiment Tracking**: Uses MLflow to log metrics, parameters, and artifacts.
- **REST API**: Serves trained models via FastAPI.
- **Containerized Deployment**: Uses Docker & Docker Compose.
- **Monitoring**: Integrated Prometheus & Grafana dashboards.
- **CI/CD**: Automated builds and deployment via GitHub Actions.

---

## 🛠️ Tech Stack
- **Python 3.10**
- **MLflow** – Model tracking and registry
- **FastAPI** – Model serving API
- **scikit-learn** – Machine learning
- **Pandas / NumPy** – Data processing
- **Docker / Docker Compose** – Containerization
- **Prometheus / Grafana** – Monitoring
- **GitHub Actions** – CI/CD automation

---

## 📂 Project Structure
├── src/
│ ├── api.py # FastAPI application
│ ├── data_preprocessing.py # Data cleaning & preprocessing
│ ├── model_training.py # Model training & MLflow logging
│ ├── utils.py # Utility functions
│ └── tests/ # Unit tests
├── artifacts/ # Model & preprocessing artifacts
├── docker-compose.yml # Multi-service container setup
├── Dockerfile.api # API container build
├── Dockerfile.train # Training container build
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## ⚡ Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/mlops-housing-clean.git
cd mlops-housing-clean

docker compose up --build

###Services:

MLflow UI → http://localhost:5000

FastAPI API → http://localhost:8000/docs

Grafana → http://localhost:3000

Prometheus → http://localhost:9090

###Train the Model:

docker compose run train

📦 CI/CD with GitHub Actions
Linting: Uses ruff and black to enforce code style.

Testing: Runs pytest.

Build & Push: Builds Docker images and pushes to registry.

Deploy: SSH to EC2 instance and runs latest containers.

Monitoring
Prometheus scrapes metrics from the API.

Grafana visualizes performance metrics.

License
This project is licensed under the MIT License.

Author

MLOPS Group

