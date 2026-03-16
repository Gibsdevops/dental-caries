from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import time
from typing import List
import uvicorn
import os
import tensorflow as tf
import traceback

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
    """Load TensorFlow Lite model using TensorFlow"""
    if not os.path.exists(model_path):
        print(f"✗ Model file not found at {model_path}")
        return None
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"✓ TFLite model loaded successfully")
        return interpreter
    except Exception as e:
        print(f"✗ Error loading TFLite model: {e}")
        traceback.print_exc()
        return None

interpreter = load_tflite_model(MODEL_PATH)

# Get input and output details
input_details = None
output_details = None
input_shape = None
input_dtype = None

if interpreter:
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = input_details[0]['shape']
        IMAGE_SIZE = (input_shape[1], input_shape[2])
        input_dtype = input_details[0]['dtype']
        
        print(f"✓ Input shape: {input_shape}")
        print(f"✓ Output shape: {output_details[0]['shape']}")
        print(f"✓ Input dtype: {input_dtype}")
    except Exception as e:
        print(f"✗ Error getting model details: {e}")
        traceback.print_exc()
        IMAGE_SIZE = (224, 224)
        input_dtype = np.float32
else:
    IMAGE_SIZE = (224, 224)
    input_dtype = np.float32

MODEL_VERSION = "1.0.0"
CLASS_LABELS = ["Caries", "Healthy"]
CONFIDENCE_THRESHOLD = 0.5


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image - matches training pipeline"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print(f"✓ Image opened, size: {image.size}")
        
        image = image.resize(IMAGE_SIZE)
        print(f"✓ Image resized to: {IMAGE_SIZE}")
        
        image_array = np.array(image, dtype=np.float32)
        print(f"✓ Image converted to array, shape: {image_array.shape}")
        
        # Normalize and preprocess for MobileNetV2
        image_array = image_array / 127.5 - 1.0
        print(f"✓ Image normalized, range: [{image_array.min():.2f}, {image_array.max():.2f}]")
        
        image_array = np.expand_dims(image_array, axis=0)
        print(f"✓ Batch dimension added, shape: {image_array.shape}")
        
        image_array = image_array.astype(input_dtype)
        print(f"✓ Image dtype converted to: {input_dtype}")
        
        return image_array
    except Exception as e:
        print(f"✗ Error in preprocess_image: {e}")
        traceback.print_exc()
        raise ValueError(f"Error preprocessing image: {str(e)}")


def run_inference(image_array: np.ndarray) -> float:
    """Run inference - returns probability of caries (0-1)"""
    if interpreter is None:
        raise ValueError("Model not loaded")
    
    try:
        print(f"Input array shape: {image_array.shape}, dtype: {image_array.dtype}")
        print(f"Input details: {input_details[0]}")
        
        interpreter.set_tensor(input_details[0]['index'], image_array)
        print(f"✓ Input tensor set")
        
        interpreter.invoke()
        print(f"✓ Model inference completed")
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"✓ Output data retrieved, shape: {output_data.shape}, dtype: {output_data.dtype}")
        print(f"✓ Output data: {output_data}")
        
        # Extract the probability (sigmoid output)
        caries_probability = float(output_data.flatten()[0])
        print(f"✓ Caries probability: {caries_probability}")
        
        return caries_probability
    except Exception as e:
        print(f"✗ Error in run_inference: {e}")
        traceback.print_exc()
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
        "framework": "TensorFlow Lite",
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
        print(f"\n=== NEW PREDICTION REQUEST ===")
        print(f"Filename: {file.filename}")
        print(f"Content type: {file.content_type}")
        
        if interpreter is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG allowed.")
        
        image_bytes = await file.read()
        print(f"Image bytes read: {len(image_bytes)} bytes")
        
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
        
        print(f"✓ Prediction successful: {prediction} ({confidence:.4f})")
        print(f"=== END PREDICTION REQUEST ===\n")
        
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
        print(f"✗ ValueError: {ve}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        traceback.print_exc()
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
                print(f"✗ Error processing file {idx}: {e}")
                traceback.print_exc()
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
        print(f"✗ Batch error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)