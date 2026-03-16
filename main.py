from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import time
from typing import List
import uvicorn
import os

# Use TensorFlow Lite Runtime
import tflite_runtime.interpreter as tflite

app = FastAPI(
    title="Dental Caries Detection API",
    description="API for detecting dental caries in dental images",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/dental_caries_model.tflite"

def load_tflite_model(model_path):
    """Load TensorFlow Lite model"""
    if not os.path.exists(model_path):
        print(f"✗ Model file not found at {model_path}")
        return None
    
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"✓ TFLite model loaded successfully")
        return interpreter
    except Exception as e:
        print(f"✗ Error loading TFLite model: {e}")
        return None

interpreter = load_tflite_model(MODEL_PATH)

# Get input and output details
input_details = None
output_details = None
input_shape = None
input_dtype = None

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    IMAGE_SIZE = (input_shape[1], input_shape[2])
    input_dtype = input_details[0]['dtype']
    
    print(f"✓ Input shape: {input_shape}")
    print(f"✓ Output shape: {output_details[0]['shape']}")
else:
    IMAGE_SIZE = (224, 224)
    input_dtype = np.float32

MODEL_VERSION = "1.0.0"
# Binary classification: 0 = Caries, 1 = Healthy
CLASS_LABELS = ["Caries", "Healthy"]
CONFIDENCE_THRESHOLD = 0.5


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image - matches training pipeline"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize and preprocess for MobileNetV2
        # This matches: tf.keras.applications.mobilenet_v2.preprocess_input(x)
        image_array = image_array / 127.5 - 1.0
        
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array.astype(input_dtype)
        return image_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def run_inference(image_array: np.ndarray) -> float:
    """Run inference - returns probability of caries (0-1)"""
    if interpreter is None:
        raise ValueError("Model not loaded")
    try:
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Extract the probability (sigmoid output)
        # output_data shape: [1, 1] for binary classification
        caries_probability = float(output_data.flatten()[0])
        
        return caries_probability
    except Exception as e:
        raise ValueError(f"Error during inference: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Dental Caries Detection API",
        "version": MODEL_VERSION,
        "docs_url": "/docs",
        "model_loaded": interpreter is not None,
        "model_type": "TensorFlow Lite - Binary Classification"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy" if interpreter is not None else "unhealthy",
        "model_loaded": interpreter is not None,
        "model_version": MODEL_VERSION,
        "timestamp": time.time()
    }


@app.get("/model-info", tags=["Model"])
async def model_info():
    if interpreter is None:
        return {"error": "Model not loaded", "model_loaded": False}
    
    return {
        "model_version": MODEL_VERSION,
        "model_type": "Binary Classification (MobileNetV2)",
        "input_size": IMAGE_SIZE,
        "classes": CLASS_LABELS,
        "classes_description": {
            "Caries": "Teeth with detected dental caries (cavities/decay)",
            "Healthy": "Healthy teeth without caries"
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "framework": "TensorFlow Lite Runtime",
        "training_data": "Dental X-rays from Zenodo dataset",
        "input_shape": list(input_shape) if input_shape is not None else None,
        "output_shape": list(output_details[0]['shape']) if output_details else None
    }


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Predict dental caries in uploaded image
    Returns probability of caries vs healthy
    """
    start_time = time.time()
    try:
        if interpreter is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG allowed.")
        
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        
        processed_image = preprocess_image(image_bytes)
        caries_probability = run_inference(processed_image)
        
        # Determine the prediction
        if caries_probability >= CONFIDENCE_THRESHOLD:
            prediction = CLASS_LABELS[0]  # Caries
            confidence = caries_probability
        else:
            prediction = CLASS_LABELS[1]  # Healthy
            confidence = 1 - caries_probability
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "filename": file.filename,
            "prediction": {
                "class": prediction,
                "confidence": round(float(confidence), 4)
            },
            "probabilities": {
                "caries": round(float(caries_probability), 4),
                "healthy": round(float(1 - caries_probability), 4)
            },
            "model_info": {
                "version": MODEL_VERSION,
                "type": "Binary Classification",
                "threshold": CONFIDENCE_THRESHOLD
            },
            "processing_time_ms": round(processing_time * 1000, 2)
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict dental caries in multiple images"""
    start_time = time.time()
    results = []
    errors = []
    
    try:
        if interpreter is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        for idx, file in enumerate(files):
            try:
                if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                    errors.append({"file_index": idx, "filename": file.filename, "error": "Invalid type"})
                    continue
                
                image_bytes = await file.read()
                processed_image = preprocess_image(image_bytes)
                caries_probability = run_inference(processed_image)
                
                if caries_probability >= CONFIDENCE_THRESHOLD:
                    prediction = CLASS_LABELS[0]
                    confidence = caries_probability
                else:
                    prediction = CLASS_LABELS[1]
                    confidence = 1 - caries_probability
                
                results.append({
                    "file_index": idx,
                    "filename": file.filename,
                    "prediction": {
                        "class": prediction,
                        "confidence": round(float(confidence), 4)
                    },
                    "probabilities": {
                        "caries": round(float(caries_probability), 4),
                        "healthy": round(float(1 - caries_probability), 4)
                    }
                })
            except Exception as e:
                errors.append({"file_index": idx, "filename": file.filename, "error": str(e)})
        
        return {
            "success": len(errors) == 0,
            "total_files": len(files),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors if errors else None,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)