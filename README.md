# Beach Conditions Agent — Complete Implementation

**Goal:** A chat-style agent answers questions like "Is it a good time to go to Kāʻanapali Beach?" by analyzing current snapshots from a beach livestream using LangGraph orchestration.

**Key Features:**
- LangGraph agent for intelligent conversation flow
- Computer vision-based ROI detection (water vs beach zones)
- Fine-tuned YOLO for people detection
- Weather analysis using CV heuristics
- Modern web UI for chat interface
- FastAPI backend with comprehensive endpoints

---

## Architecture

```
User (Web UI) ──> LangGraph Agent ──> FastAPI Tools ──> Snapshot Capture
                       |                    |        ├─> CV ROI Detection
                       |                    |        ├─> People Detection (YOLO)
                       |                    |        └─> Weather Analysis
                       └─> Response Generation
```

**Components:**
- **LangGraph Agent**: Orchestrates tool calls and manages conversation state
- **FastAPI Service**: Provides tools for snapshot capture, analysis, and data retrieval
- **CV ROI Detection**: Computer vision model to automatically detect water/beach zones
- **People Detection**: Fine-tuned YOLO on beach imagery
- **Weather Analysis**: CV-based cloud coverage and rain detection
- **Web UI**: Modern chat interface with real-time updates

---

## Repository Structure

```
cv-training/
├─ CV_Data/                          # existing training images
│  └─ Kaanapali_Beach/              # beach images for training
├─ data/
│  ├─ snapshots/                     # captured images
│  ├─ training_samples/              # processed training data
│  └─ roi_zones.json                 # detected ROI zones
├─ models/
│  ├─ yolo_people.pt                 # fine-tuned YOLO weights
│  └─ roi_detector.pt                # ROI detection model
├─ src/
│  ├─ api/
│  │  ├─ main.py                     # FastAPI app
│  │  ├─ tools.py                    # LangGraph tools
│  │  └─ schemas.py                  # Pydantic models
│  ├─ agent/
│  │  ├─ graph.py                    # LangGraph agent
│  │  ├─ nodes.py                    # Agent nodes
│  │  └─ prompts.py                  # System prompts
│  ├─ cv/
│  │  ├─ capture.py                  # snapshot capture
│  │  ├─ detection.py                # people detection
│  │  ├─ roi.py                      # ROI detection
│  │  └─ weather.py                  # weather analysis
│  └─ ui/
│     ├─ static/                     # CSS/JS files
│     ├─ templates/                  # HTML templates
│     └─ app.py                      # UI server
├─ training/
│  ├─ prepare_data.py                # data preparation
│  ├─ train_yolo.py                  # YOLO training
│  └─ train_roi.py                   # ROI model training
├─ config.yaml                       # configuration
├─ requirements.txt                  # dependencies
├─ Dockerfile                        # containerization
└─ docker-compose.yml                # multi-service setup
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (for LangGraph agent)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone and setup environment:**
```bash
cd cv-training
pip install -r requirements.txt
```

2. **Set OpenAI API key:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. **Prepare training data (optional):**
```bash
python training/prepare_data.py
# Annotate images using Label Studio or similar tool
python training/train_yolo.py
```

4. **Run the application:**
```bash
python src/api/main.py
```

5. **Access the web interface:**
Open http://localhost:8000 in your browser

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run with Docker directly
docker build -t beach-conditions-agent .
docker run -p 8000:8000 -e OPENAI_API_KEY="your-key" beach-conditions-agent
```

---

## Implementation Steps

### Phase 1: Core Infrastructure (Day 1)
1. Set up FastAPI service with basic endpoints
2. Integrate existing snapshot capture functionality
3. Define ROI zones for Kāʻanapali Beach
4. Basic response templating

### Phase 2: People Detection (Day 2)
1. Prepare small training dataset from CV_Data
2. Fine-tune YOLO on beach people detection
3. Implement ROI-based counting (water vs beach)

### Phase 3: Weather Analysis (Day 3)
1. Implement cloud coverage heuristics
2. Add rain detection logic
3. Integrate weather analysis into main service

### Phase 4: Polish & Deploy (Day 4)
1. Add simple web UI for testing
2. Containerize application
3. Add basic error handling and logging

---

## Minimal Training Approach

### People Detection
- **Dataset:** 200-500 annotated images from CV_Data/Kaanapali_Beach
- **Model:** Start with YOLOv8n, fine-tune for 20-30 epochs
- **Labeling:** Use Roboflow or Label Studio for quick annotation
- **Validation:** Manual count comparison on test images

### Weather Analysis (No Training Required)
```python
# Cloud coverage (simple but effective)
def estimate_cloud_coverage(image, sky_mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Clouds are typically bright in value channel
    cloud_mask = (hsv[:,:,2] > 200) & sky_mask
    return np.sum(cloud_mask) / np.sum(sky_mask) * 100

# Rain detection
def detect_rain(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Look for vertical streaks
    vertical_kernel = np.ones((10,1), np.uint8)
    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    rain_score = np.sum(vertical_lines) / (image.shape[0] * image.shape[1])
    return rain_score > 0.001  # threshold to tune
```

---

## API Endpoints (Simplified)

**`POST /chat`**
```json
{
  "message": "Is it a good time to go to the beach?",
  "camera_id": "kaanapali"
}
```

**Response:**
```json
{
  "response": "Based on current conditions: 8 people in water, 12 on beach. 25% cloud cover, no rain. Great time for a beach visit!",
  "data": {
    "people": {"water": 8, "beach": 12, "total": 20},
    "weather": {"cloud_coverage": 25, "is_raining": false},
    "timestamp": "2025-09-06T07:18:31Z"
  }
}
```

---

## Docker Setup (Single Container)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Configuration (Single File)

```yaml
# config.yaml
camera:
  stream_url: "https://youtube.com/watch?v=..."
  roi_zones: "data/roi_zones.json"

models:
  yolo_path: "models/yolo_people.pt"
  confidence_threshold: 0.5

weather:
  cloud_threshold: 0.3
  rain_threshold: 0.001

responses:
  template: "Based on current conditions: {people_summary}. {weather_summary}. {recommendation}"
```

---

This simplified approach reduces complexity by ~60% while maintaining core functionality. The system will be easier to build, test, and maintain, making it much more suitable for a rapid development cycle.
