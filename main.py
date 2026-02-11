from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from pathlib import Path
import traceback
from preprocess import preprocess_image, extract_fusion_features

app = FastAPI(title="Vehicle Classification API")

# CORS - Must be before routes
# Update setelah deploy frontend untuk security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://*.vercel.app",  # Vercel preview & production
        "*"  # TEMPORARY - ganti dengan domain spesifik setelah deploy
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Load models saat startup
MODEL_DIR = Path("models")

try:
    svm_model = joblib.load(MODEL_DIR / "svm_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    pca = joblib.load(MODEL_DIR / "pca.joblib")
    feature_dimensions = joblib.load(MODEL_DIR / "feature_dimensions.joblib")
    model_info = joblib.load(MODEL_DIR / "model_info.joblib")
    
    print("✅ All models loaded successfully!")
    print(f"Classes: {model_info['classes']}")
    print(f"Image size: {feature_dimensions['img_size']}")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    svm_model = None

@app.get("/")
def health_check():
    """Health check endpoint"""
    if svm_model is None:
        return {"status": "error", "message": "Models not loaded"}
    return {
        "status": "ok",
        "message": "Vehicle Classification API is running",
        "model_loaded": True
    }

@app.get("/model-info")
def get_model_info():
    """Get model information"""
    if svm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": model_info['classes'],
        "train_accuracy": float(model_info['train_accuracy']),
        "test_accuracy": float(model_info['test_accuracy']),
        "total_features_before_pca": int(model_info['total_feat_before_pca']),
        "total_features_after_pca": int(model_info['total_feat_after_pca']),
        "variance_retained": float(model_info['variance_retained']),
        "image_size": feature_dimensions['img_size']
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict vehicle class dari uploaded image
    """
    # Validasi model
    if svm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validasi file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/jfif"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Read image
        contents = await file.read()
        
        # Preprocess (resize, blur, normalize, grayscale)
        img_size = tuple(feature_dimensions['img_size'])
        img_gray = preprocess_image(contents, target_size=img_size)
        
        # Extract features (HOG+Sobel, LBP)
        features_dict = extract_fusion_features(img_gray)
        hog_sobel_features = features_dict['hog_sobel']
        lbp_features = features_dict['lbp']
        
        # Fusion: concatenate HOG+Sobel dengan LBP (sama seperti notebook)
        fusion_features = np.concatenate([hog_sobel_features, lbp_features])
        
        # Reshape untuk single prediction
        fusion_features = fusion_features.reshape(1, -1)
        
        # Validasi dimensi features
        expected_dim = feature_dimensions['fusion']
        if fusion_features.shape[1] != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch. Expected: {expected_dim}, "
                f"Got: {fusion_features.shape[1]}"
            )
        
        # Scale features
        scaled_features = scaler.transform(fusion_features)
        
        # PCA transform
        pca_features = pca.transform(scaled_features)
        
        # Predict
        prediction = svm_model.predict(pca_features)[0]
        
        # Convert numpy types to Python native types
        prediction = int(prediction)
        
        # Get probability/confidence (jika SVM pakai probability=True)
        try:
            probabilities = svm_model.predict_proba(pca_features)[0]
            confidence = float(max(probabilities))
            
            # Get top 3 predictions
            top_3_idx = np.argsort(probabilities)[::-1][:3]
            top_3_predictions = [
                {
                    "class": str(model_info['classes'][int(idx)]),
                    "confidence": float(probabilities[idx])
                }
                for idx in top_3_idx
            ]
        except:
            # Jika SVM tidak pakai probability
            confidence = None
            top_3_predictions = None
        
        return {
            "success": True,
            "prediction": {
                "class": str(model_info['classes'][prediction]),  # Ensure string
                "confidence": confidence,
            },
            "top_3": top_3_predictions,
            "filename": file.filename
        }
        
    except Exception as e:
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"❌ Error during prediction: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        
        # Return detailed error for debugging
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction failed",
                "error": str(e),
                "type": type(e).__name__
            }
        )

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict multiple images at once
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per request"
        )
    
    results = []
    for file in files:
        try:
            # Reuse predict logic
            result = await predict(file)
            results.append(result)
        except Exception as e:
            results.append({
                "success": False,
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total": len(files),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)