# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (gcc for potential package builds, curl for health checks)
# Define and create directories in the same layer where ENV is used
# Set environment variables early for use in this layer
ENV LOG_DIR=/app/logs \
    MODEL_DIR=/app/models \
    REPORTS_DIR=/app/reports \
    # Default for Docker connecting to host. Override at runtime if needed.
    MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
    MODEL_NAME=CaliforniaHousingRegressor \
    # Use MODEL_ALIAS to match api.py
    MODEL_ALIAS=staging \
    PYTHONPATH="/app"

# Create necessary directories using the defined ENV variables
# Also create directories that might be mounted or used for reports
RUN apt-get update && \
    apt-get install -y gcc curl && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p "$LOG_DIR" "$MODEL_DIR/preprocessing" "$REPORTS_DIR" "$REPORTS_DIR/eda_report" /app/data /app/scripts

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
# Copy the source code directory
# Copy the rest of the application code

COPY . .

# --- Copy other necessary directories and files ---
# These commands will copy directories if they exist in the build context (your local project folder).
# If they don't exist locally, the build might fail. Docker Compose volumes will handle runtime needs.
# It's generally okay to copy them if they exist, but ensure they are in your project repo.
COPY src/ ./src/
COPY data/ ./data/
COPY reports/ ./reports/
# --- End copying other directories ---

# If you have config files or other items in the project root, copy them too
# Example: COPY config.yaml .
# Example: COPY README.md .
# Example: COPY *.py . # Copy any .py files directly in the project root (be careful with this if it includes large files)

# Expose the port the app runs on
EXPOSE 8000

# Health check using curl
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application by default
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]