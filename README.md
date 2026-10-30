# ElderCare AI — Fall Detection & Elderly Monitoring System

An AI-assisted elderly care monitoring platform that combines pose estimation, temporal deep learning, gait analysis, and caregiver alerts to identify potential falls and monitor daily activities.

## Overview

ElderCare AI is designed to process video footage, extract human pose information, recognize activities, and detect potential fall events. The system combines a machine-learning pipeline with a web dashboard, database, and notification services.

The project explores how computer vision and temporal modelling can support elderly safety monitoring.

> **Project status:** The core web application and ML pipeline are described as implemented. The mobile application integration, automated testing, and deployment workflows require further verification.

## Key Features

- **Pose estimation:** Extracts human skeletal keypoints from video frames.
- **Fall detection:** Classifies observations as fall or non-fall events.
- **Activity recognition:** Recognizes seven activity categories.
- **Gait analysis:** Estimates fall risk using movement-related features.
- **Alert generation:** Detects configured events such as falls, SOS gestures, prolonged inactivity, and elevated fall risk.
- **Caregiver notifications:** Supports email and Firebase Cloud Messaging integrations.
- **Monitoring dashboard:** Displays patient information, activity summaries, session results, and alert history.
- **Data persistence:** Uses PostgreSQL to store session information, predictions, and alerts.

## System Architecture

The project is organized into the following components:

```text
ElderCareProject/
├── eldercare_v2/
│   ├── backend/          # FastAPI application
│   ├── frontend/         # React dashboard
│   ├── database/         # PostgreSQL schema
│   └── docker-compose.yml
├── pipeline/
│   ├── inference.py      # Video inference orchestration
│   ├── pose_extractor.py # Pose estimation
│   ├── gait_analyzer.py  # Movement and fall-risk analysis
│   └── extract_pipeline.py
├── models/
│   ├── tcn_lstm_model.py
│   ├── binary_fall_train.py
│   ├── activity_train.py
│   ├── binary_fall_best.pth
│   └── activity_best.pth
├── alerts/
│   ├── sos_detector.py
│   ├── email_alert.py
│   └── firebase_alert.py
├── configs/
│   └── config.py
├── datasets/
│   ├── raw/
│   └── processed/
├── eldercare_flutter_app/
├── run.py
└── AGENTS.md
`

## Technology Stack

| Component | Technologies |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Frontend | React 18, Vite, Tailwind CSS |
| Database | PostgreSQL, SQLAlchemy |
| ML and deep learning | PyTorch, scikit-learn, NumPy, Pandas |
| Pose estimation | YOLOv8 Pose, MediaPipe |
| Video processing | OpenCV |
| Model architecture | Temporal Convolutional Network (TCN) and BiLSTM |
| Visualizations | Recharts |
| State management | Zustand |
| Infrastructure | Docker, Docker Compose, Redis |
| Notifications | SMTP email, Firebase Cloud Messaging |
| Mobile application | Flutter |

## Machine Learning Pipeline

The processing workflow is intended to follow these stages:

1. **Video input:** A user uploads a video through the application.
2. **Pose extraction:** The video-processing pipeline detects a person and extracts skeletal landmarks.
3. **Sequence preparation:** Pose observations are organized into temporal sequences. The described configuration uses 30-frame windows.
4. **Model inference:** The trained models predict fall events and recognize activities.
5. **Gait analysis:** Movement-related features are used to estimate potential fall risk.
6. **Alert evaluation:** The alert module checks predictions and configured thresholds.
7. **Notification delivery:** Eligible events can trigger email or Firebase notifications.
8. **Storage and visualization:** Session data, predictions, and alert history are stored and presented in the dashboard.

### Pose Estimation

The documented pipeline combines YOLOv8 Pose and MediaPipe. YOLO-based pose detection is described as providing 17 COCO keypoints, while MediaPipe provides 33 pose landmarks.


### Temporal Modelling

The project includes a TCN–BiLSTM architecture.

- **TCN:** Learns temporal patterns from sequences using temporal convolutions.
- **BiLSTM:** Processes sequence information in both forward and backward directions.
- **Fall classification:** Predicts fall versus non-fall events.
- **Activity classification:** Predicts one of the configured activity categories.

## Model Performance

The supplied project overview reports the following figures:

| Model | Task | Reported performance |
|---|---|---|
| `binary_fall_best.pth` | Binary fall detection | 95%+ accuracy |
| `activity_best.pth` | Seven-class activity recognition | Not specified |

**Evaluation note:** These figures are reported in the existing project documentation and should not be treated as independently reproduced results. Before publishing performance claims, verify the evaluation dataset, train/test separation, class distribution, metric calculations, and model checkpoint.

For a reproducible evaluation, report accuracy, precision, recall, F1-score, and the confusion matrix where appropriate.

## Alert System

The alert module describes five event categories:

| Alert | Trigger described in the project |
|---|---|
| `FALL` | Fall confidence of 0.75 or higher |
| `WAVE_SOS` | At least three wave gestures within 10 seconds |
| `PROLONGED_SLEEP` | Continuous sleeping exceeding 15 hours |
| `PROLONGED_INACTIVITY` | No detected movement for more than two hours |
| `FALL_RISK` | Gait-risk score of 0.70 or higher |


## Backend API

The FastAPI application exposes endpoints for authentication, video processing, patient management, dashboard statistics, and alert handling.

### Authentication

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/auth/login` | Authenticate a user |
| GET | `/api/auth/me` | Retrieve the current user |

### Video Processing

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/sessions/upload` | Upload a video for processing |
| GET | `/api/sessions/{id}` | Retrieve session details |
| GET | `/api/sessions/{id}/predictions` | Retrieve predictions |
| GET | `/api/sessions/{id}/video/skeleton` | Retrieve the skeleton-overlay video |

### Patient Management

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/patients` | List patients |
| POST | `/api/patients` | Create a patient |
| GET | `/api/patients/{id}` | Retrieve patient details |

### Dashboard and Alerts

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/dashboard/stats` | Retrieve dashboard statistics |
| GET | `/api/dashboard/activity-distribution` | Retrieve activity summaries |
| GET | `/api/dashboard/recent-alerts` | Retrieve recent alerts |
| GET | `/api/alerts` | Retrieve alert history |
| POST | `/api/alerts/{id}/acknowledge` | Acknowledge an alert |

For a running instance, interactive API documentation is expected at `/docs`.

## Local Setup

### Prerequisites

- Python 3.9 or a compatible version supported by the dependencies
- Node.js and npm
- Docker and Docker Compose, or a local PostgreSQL installation
- Git
- Optional CUDA-compatible GPU for accelerated inference

### Option 1: Docker Compose

From the application directory:

```bash
cd eldercare_v2
docker compose up --build
```

The documented local service addresses are:

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`
- PostgreSQL: port `5432`

These addresses and service configurations should be checked against `docker-compose.yml`.

### Option 2: Manual Setup

**Backend**

```bash
cd eldercare_v2/backend
python -m venv .venv
```

Activate the virtual environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies and start the API:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend**

Open another terminal:

```bash
cd eldercare_v2/frontend
npm install
npm run dev
```

Start PostgreSQL separately and initialize the database using the schema and database instructions provided in the repository.

## Environment Configuration


```dotenv
DATABASE_URL=
SECRET_KEY=
ALERT_EMAIL=
ALERT_PASSWORD=
FIREBASE_CREDENTIALS_PATH=
FCM_DEVICE_TOKENS=
MODEL_ROOT=
CONFIDENCE_THRESHOLD=0.35
```


## Data Management

The project separates raw videos from processed pose data:

- `datasets/raw/`: source video files
- `datasets/processed/`: extracted pose sequences and related CSV files


## Current Limitations and Future Work

The supplied project overview identifies the following areas for further work:

- Integrating the Flutter mobile application with the backend API.
- Adding unit tests and end-to-end integration tests.
- Implementing continuous integration and automated checks.
- Improving API error handling and edge-case coverage.
- Verifying model performance on independent, representative datasets.
- Evaluating false positives, false negatives, and performance across different camera conditions.
- Testing alert delivery, notification cooldowns, and recovery from service failures.
- Reviewing privacy, access control, data retention, and security practices.

This system should be treated as an experimental monitoring aid, not a replacement for professional medical assessment or emergency services.

## Reproducibility and Contributions

- Dataset sources and preprocessing procedures.
- Model architecture and hyperparameters.
- Training and validation configuration.
- Evaluation scripts and metric definitions.
- Software dependencies and hardware used.
- Random seeds and checkpoint provenance, where applicable.


## License and Attribution



**Project:** ElderCare AI — Fall Detection & Elderly Monitoring System  
**Version:** 2.0, if confirmed by the repository  
**Primary focus:** Pose-based fall detection, temporal activity recognition, and caregiver alerting
