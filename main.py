from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import time
from typing import List
import uvicorn
import tensorflow as tf
import os

app = FastAPI(
    title="Dental Caries Detection API",
    description="API for detecting dental caries in dental images",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained TensorFlow Lite model
MODEL_PATH = "models/dental_caries_model.tflite"  # Change to your .tflite model path

def load_tflite_model(model_path):
    """Load TensorFlow Lite model"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"✓ TFLite model loaded successfully from {model_path}")
        return interpreter
    except Exception as e:
        print(f"✗ Error loading TFLite model: {e}")
        return None

interpreter = load_tflite_model(MODEL_PATH)

# Get input and output details
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Extract input shape (usually [1, height, width, channels])
    input_shape = input_details[0]['shape']
    IMAGE_SIZE = (input_shape[1], input_shape[2])  # (height, width)
    input_dtype = input_details[0]['dtype']
    
    print(f"✓ Input shape: {input_shape}")
    print(f"✓ Input dtype: {input_dtype}")
    print(f"✓ Image size: {IMAGE_SIZE}")
else:
    IMAGE_SIZE = (224, 224)
    input_dtype = np.float32

MODEL_VERSION = "1.0.0"
CLASS_LABELS = ["No Caries", "Caries"]  # Adjust based on your model's output classes


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for TFLite model inference
    """
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize to model input size
        image = image.resize(IMAGE_SIZE)
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array / 255.0
        
        # If your model was trained with ImageNet normalization, uncomment this:
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image_array = (image_array - mean) / std
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        # Ensure correct dtype for TFLite
        image_array = image_array.astype(input_dtype)
        
        return image_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def run_inference(image_array: np.ndarray) -> np.ndarray:
    """
    Run TensorFlow Lite model inference
    """
    if interpreter is None:
        raise ValueError("Model not loaded. Please check the model path.")
    
    try:
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], image_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get confidence scores (usually shape [1, num_classes])
        confidence_scores = output_data[0]
        
        return confidence_scores
    except Exception as e:
        raise ValueError(f"Error during inference: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Dental Caries Detection API",
        "version": MODEL_VERSION,
        "docs_url": "/docs",
        "model_loaded": interpreter is not None,
        "model_type": "TensorFlow Lite"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if interpreter is not None else "unhealthy",
        "model_loaded": interpreter is not None,
        "model_version": MODEL_VERSION,
        "model_type": "TensorFlow Lite",
        "timestamp": time.time()
    }


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get model information"""
    if interpreter is None:
        return {
            "error": "Model not loaded",
            "model_loaded": False
        }
    
    return {
        "model_version": MODEL_VERSION,
        "input_size": IMAGE_SIZE,
        "classes": CLASS_LABELS,
        "framework": "TensorFlow Lite",
        "description": "Dental caries detection model trained on annotated dental images from Zenodo dataset",
        "model_path": MODEL_PATH,
        "input_shape": list(input_shape),
        "input_dtype": str(input_dtype),
        "output_shape": list(output_details[0]['shape'])
    }


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Predict dental caries in uploaded image
    
    Returns:
    - success: Boolean indicating if prediction was successful
    - predictions: List of predictions with class and confidence
    - confidence_scores: Raw confidence scores for each class
    - processing_time: Time taken to process the image
    - model_version: Version of the model used
    """
    start_time = time.time()
    
    try:
        if interpreter is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG and PNG images are allowed."
            )
        
        # Read image bytes
        image_bytes = await file.read()
        
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Run inference
        confidence_scores = run_inference(processed_image)
        
        # Build predictions list
        predictions = []
        for idx, confidence in enumerate(confidence_scores):
            predictions.append({
                "class": CLASS_LABELS[idx],
                "confidence": float(confidence)
            })
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Get the top prediction
        top_prediction = predictions[0]
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "top_prediction": top_prediction,
            "predictions": predictions,
            "confidence_scores": {
                CLASS_LABELS[i]: float(confidence_scores[i]) 
                for i in range(len(CLASS_LABELS))
            },
            "processing_time": round(processing_time, 4),
            "model_version": MODEL_VERSION,
            "filename": file.filename
        }
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict dental caries in multiple images
    
    Returns:
    - success: Boolean indicating if all predictions were successful
    - results: List of predictions for each image
    - processing_time: Total time taken
    """
    start_time = time.time()
    results = []
    errors = []
    
    try:
        if interpreter is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        for idx, file in enumerate(files):
            try:
                if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                    errors.append({
                        "file_index": idx,
                        "filename": file.filename,
                        "error": "Invalid file type"
                    })
                    continue
                
                image_bytes = await file.read()
                processed_image = preprocess_image(image_bytes)
                confidence_scores = run_inference(processed_image)
                
                predictions = []
                for cidx, confidence in enumerate(confidence_scores):
                    predictions.append({
                        "class": CLASS_LABELS[cidx],
                        "confidence": float(confidence)
                    })
                
                predictions.sort(key=lambda x: x["confidence"], reverse=True)
                
                results.append({
                    "file_index": idx,
                    "filename": file.filename,
                    "top_prediction": predictions[0],
                    "predictions": predictions,
                    "confidence_scores": {
                        CLASS_LABELS[i]: float(confidence_scores[i]) 
                        for i in range(len(CLASS_LABELS))
                    }
                })
            
            except Exception as e:
                errors.append({
                    "file_index": idx,
                    "filename": file.filename,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        return {
            "success": len(errors) == 0,
            "total_files": len(files),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
            "processing_time": round(processing_time, 4),
            "model_version": MODEL_VERSION
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)