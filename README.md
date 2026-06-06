# NASA Satellite Telemetry Anomaly Detector 🛰️

An unsupervised time-series anomaly detection system built using a **Temporal Convolutional Network (TCN) Autoencoder** and an interactive monitoring dashboard powered by **Streamlit**. The system processes multivariate NASA satellite telemetry streams, learns normal operational patterns, and automatically flags anomalous behaviors using advanced statistical thresholding techniques.

---

##  Project Overview

Modern satellite infrastructure continuously generates large volumes of telemetry data from onboard sensors. Detecting abnormal system behavior manually is difficult and time-consuming.

This project leverages deep learning and statistical anomaly detection to:

* Learn normal telemetry behavior without labeled anomaly data.
* Detect unusual system states automatically.
* Visualize anomalies through an interactive dashboard.
* Generate reproducible engineering reports.

The solution combines:

* Temporal Convolutional Network (TCN) Autoencoder
* Exponential Moving Average (EMA) Smoothing
* Peak-Over-Threshold (POT) Statistical Modeling
* Streamlit Dashboard
* Dockerized Deployment Pipeline

---

#  System Architecture & Data Flow

```text
Raw NASA Data
      │
      ▼
Scaling & Windowing
      │
      ▼
TCN Autoencoder Training
      │
      ▼
Reconstruction Error Calculation
      │
      ▼
EMA Score Smoothing
      │
      ▼
Thresholding
(POT + Percentile)
      │
      ▼
Anomaly Detection
      │
      ▼
Streamlit Dashboard Visualization
```

---

##  Pipeline Components

### 1. Data Preprocessing

Telemetry signals from the NASA SMAP dataset are:

* Downloaded automatically
* Normalized using feature scaling
* Converted into sliding windows of length 100
* Saved as model-ready tensors

---

### 2. TCN Autoencoder Training

A Temporal Convolutional Network Autoencoder is trained using:

* Dilated causal convolutions
* Residual connections
* Mean Squared Error (MSE) reconstruction loss

The model learns compressed latent representations of normal system behavior.

---

### 3. Inference & Scoring

After training:

* Test sequences pass through the network.
* Reconstruction errors are calculated.
* Error timelines are generated.

To reduce noise, scores are smoothed using:

```math
EMA_t = αx_t + (1-α)EMA_{t-1}
```

---

### 4. Statistical Thresholding

Two anomaly thresholding methods are implemented:

#### Fixed Percentile Thresholding

Flags observations exceeding a predefined percentile.

#### Peak-Over-Threshold (POT)

Uses a Generalized Pareto Distribution (GPD) to model extreme reconstruction errors and detect rare events.

---

#  Technology Stack

## Machine Learning

* PyTorch
* Temporal Convolutional Networks (TCN)
* Autoencoders

## Statistical Analysis

* Peak Over Threshold (POT)
* Generalized Pareto Distribution
* Exponential Moving Average (EMA)

## Dashboard & Visualization

* Streamlit
* Plotly

## Data Processing

* NumPy
* Pandas
* Scikit-learn

## Containerization

* Docker
* Docker Compose

---

#  Repository Structure

```text
project-root/
│
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── submission.json
│
├── scripts/
│   ├── preprocess_data.py
│   ├── train.py
│   └── evaluate.py
│
├── app/
│   └── main.py
│
├── data/
│
├── models/
│
├── results/
│
└── docs/
    └── TCN_vs_LSTM.md
```

---

##  Directory Explanation

| Directory/File               | Description                              |
| ---------------------------- | ---------------------------------------- |
| `docker-compose.yml`         | Docker orchestration configuration       |
| `Dockerfile`                 | Application container build instructions |
| `.env.example`               | Environment variable template            |
| `submission.json`            | Automated evaluation configuration       |
| `scripts/preprocess_data.py` | Data preprocessing pipeline              |
| `scripts/train.py`           | TCN model training script                |
| `scripts/evaluate.py`        | Thresholding and anomaly scoring         |
| `app/main.py`                | Streamlit dashboard                      |
| `data/`                      | Raw and processed telemetry data         |
| `models/`                    | Saved PyTorch model checkpoints          |
| `results/`                   | Evaluation outputs and reports           |
| `docs/TCN_vs_LSTM.md`        | Technical comparison document            |

---

#  Quickstart Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/prabhasupriya/time_Series_Anomaly_detector_with_tcnautoencoder_and_streamlit.git

cd time_Series_Anomaly_detector_with_tcnautoencoder_and_streamlit
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Data Pipeline

Generate processed datasets:

```bash
python scripts/preprocess_data.py
```

---

## 4️⃣ Train TCN Autoencoder

```bash
python scripts/train.py
```

---

## 5️⃣ Evaluate & Generate Thresholds

```bash
python scripts/evaluate.py
```

---

# 🐳 Docker Deployment

Build and launch the entire application stack:

```bash
docker-compose up --build
```

---

## Access Dashboard

Open your browser:

```text
http://localhost:8501
```

---

# 📊 Dashboard Features

The Streamlit monitoring interface includes four primary telemetry investigation modules.

---

##  Signal Explorer

* Interactive channel selection
* Multi-variable telemetry browsing
* Time-series exploration

---

##  Reconstruction Overlay

Visual comparison between:

* Original sensor values
* Autoencoder reconstructions

Used to identify deviations from learned behavior.

---

##  Anomaly Score Timeline

Displays:

* Raw reconstruction loss
* Smoothed EMA loss
* Threshold boundaries
* Detected anomaly regions

---

## 🔍 Channel Contribution Breakdown

Provides sensor-level attribution explaining:

* Which channels triggered anomalies
* Relative contribution strengths
* Root cause indicators

---

# 📄 Automated Report Generation

At the bottom of the dashboard:

### Generate Full Report

Clicking this button automatically creates:

```text
results/streamlit_report.json
```

This file contains:

* Evaluation metrics
* Threshold statistics
* Anomaly counts
* Verification artifacts

---

# 🔬 Why TCN Instead of LSTM?

## Massive Parallelization

Unlike recurrent architectures:

```text
t₁ → t₂ → t₃ → t₄
```

TCNs process entire sequences simultaneously.

Benefits:

* Faster training
* Better GPU utilization
* Improved scalability

---

## Stable Gradient Flow

Residual skip connections help avoid:

* Vanishing gradients
* Exploding gradients

This improves optimization stability for deep temporal models.

---

## Controlled Historical Context

TCNs use dilated convolutions with predictable receptive fields:

```math
R = 1 + 2 \cdot \sum (K - 1) \cdot D
```

Where:

* R = Receptive Field
* K = Kernel Size
* D = Dilation Factor

This enables long-range dependency modeling without recurrent memory bottlenecks.

---

# 📈 Future Enhancements

* Real-time telemetry streaming
* Satellite health scoring engine
* Attention-enhanced TCN architectures
* Transformer-based anomaly detection comparison
* Multi-satellite monitoring support
* Alert notification pipelines
* Cloud deployment with Kubernetes

---
## youtude video -[click here to watch](https://youtu.be/pNsjPvlJ76w)

# 🧑‍💻 Author

### Bandaru Prabha Supriya

* AIML Undergraduate
* Machine Learning Enthusiast
* Data Science Practitioner
* Full-Stack AI Developer

---

# ⭐ Support

If you found this project useful, consider giving the repository a **Star ⭐** on GitHub.

Your support helps improve future AI, MLOps, and anomaly detection projects.
