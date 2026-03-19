#!/usr/bin/env python3
"""
CNIE Classification API Server
Clean implementation with feedback collection for retraining.
"""

import io
import base64
import json
import time
from pathlib import Path
import hashlib
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from inference_engine_3class import CNIEClassifier3Class, get_3class_classifier
from feedback_system import get_feedback_collector


# Models
class PredictionResponse(BaseModel):
    success: bool
    predicted_class: str
    confidence: float
    all_scores: dict
    inference_time_ms: float


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: str


class HistoryItem(BaseModel):
    id: str
    timestamp: str
    thumbnail: str  # base64
    predicted_class: str
    confidence: float
    status: str  # pending, flagged, uploading, uploaded, failed


# Server State
@dataclass
class ServerState:
    classifier: Optional[CNIEClassifier3Class] = None
    model_version: str = "3.0"
    start_time: float = field(default_factory=time.time)
    total_requests: int = 0


state = ServerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context."""
    print("=" * 60)
    print("CNIE Classification API Server Starting...")
    print("=" * 60)
    
    try:
        model_path = Path.home() / 'retin-verify/models/classification/cnie_classifier_3class_v2.pth'
        print(f"Loading 3-class model from: {model_path}")
        state.classifier = CNIEClassifier3Class(model_path=model_path, device='auto')
        state.model_version = "3class"
        print(f"✓ 3-class model loaded successfully")
        print(f"✓ Classes: {state.classifier.CLASS_NAMES}")
        print(f"✓ Device: {state.classifier.device}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        state.classifier = None
    
    print("=" * 60)
    yield
    
    print("Shutting down...")


# Create App
app = FastAPI(
    title="CNIE Classification API",
    description="CNIE Front/Back classification with feedback",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main application."""
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text())
    return HTMLResponse(content="<h1>CNIE Classification API</h1><p>Frontend not found</p>")


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy" if state.classifier else "unhealthy",
        "model_loaded": state.classifier is not None,
        "uptime_seconds": time.time() - state.start_time,
        "total_requests": state.total_requests
    }


@app.get("/info")
async def info():
    """Model information."""
    if not state.classifier:
        raise HTTPException(503, "Model not loaded")
    
    return {
        "model_path": str(state.classifier.model_path),
        "model_size_mb": state.classifier.model_path.stat().st_size / 1024 / 1024,
        "num_classes": len(state.classifier.CLASS_NAMES),
        "classes": state.classifier.CLASS_NAMES,
        "device": str(state.classifier.device),
        "input_size": state.classifier.input_size
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Classify an image."""
    if not state.classifier:
        raise HTTPException(503, "Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        start = time.time()
        result = state.classifier.predict(image, return_all_scores=True)
        inference_time = (time.time() - start) * 1000
        
        state.total_requests += 1
        
        return PredictionResponse(
            success=True,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            all_scores=result['all_scores'],
            inference_time_ms=inference_time
        )
    
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: Request):
    """Submit feedback for retraining."""
    try:
        data = await request.json()
        
        # Decode image
        image_data = base64.b64decode(data['image_base64'])
        
        # Determine correct class
        correct_class = data.get('correct_class')
        if not correct_class:
            # If not provided, assume the other class
            predicted = data['predicted_class']
            correct_class = 'cnie_back' if predicted == 'cnie_front' else 'cnie_front'
        
        # Submit to collector
        collector = get_feedback_collector()
        result = collector.submit_feedback(
            image_data=image_data,
            predicted_class=data['predicted_class'],
            predicted_confidence=data['predicted_confidence'],
            is_correct=False,
            correct_class=correct_class,
            notes=data.get('notes', 'Flagged via UI')
        )
        
        return FeedbackResponse(
            success=True,
            message="Image saved for retraining",
            feedback_id=result['feedback_id']
        )
    
    except Exception as e:
        raise HTTPException(400, f"Failed to save feedback: {str(e)}")


@app.post("/feedback_no_card")
async def submit_no_card_feedback(request: Request):
    """Submit image as 'no_card' sample for 3-class training."""
    try:
        data = await request.json()
        image_data = base64.b64decode(data['image_base64'])
        
        # Save to feedback_data_3class/no_card/
        no_card_dir = Path.home() / 'retin-verify/apps/classification/feedback_data_3class/no_card'
        no_card_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        image_hash = hashlib.md5(image_data).hexdigest()[:12]
        filename = f"{timestamp}_{image_hash}.jpg"
        
        image_path = no_card_dir / filename
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        return {
            'success': True,
            'message': 'Saved as no_card sample',
            'path': str(image_path)
        }
    except Exception as e:
        raise HTTPException(400, f"Failed to save: {str(e)}")


@app.get("/feedback/stats")
async def feedback_stats():
    """Get feedback collection statistics."""
    collector = get_feedback_collector()
    return collector.get_statistics()


@app.post("/retrain/prepare")
async def prepare_retraining():
    """Prepare retraining dataset from feedback."""
    try:
        collector = get_feedback_collector()
        stats = collector.get_statistics()
        
        if not stats['retraining_recommended']:
            return {
                "success": False,
                "message": f"Need 10+ misclassified images. Current: {stats['misclassified']}"
            }
        
        dataset_dir = collector.prepare_retraining_dataset()
        
        return {
            "success": True,
            "dataset_path": str(dataset_dir),
            "train_count": len(list(dataset_dir.glob('train/*/*.jpg'))),
            "val_count": len(list(dataset_dir.glob('val/*/*.jpg'))),
            "message": "Dataset ready for retraining"
        }
    
    except Exception as e:
        raise HTTPException(500, f"Failed to prepare dataset: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
