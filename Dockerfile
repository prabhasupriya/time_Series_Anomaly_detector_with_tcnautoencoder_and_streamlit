FROM python:3.9-slim

# Install only what is strictly necessary for building dependencies and healthchecks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copying requirements or app code files explicitly
COPY . /workspace

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    scipy==1.11.1 \
    streamlit==1.25.0 \
    plotly==5.15.0

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]