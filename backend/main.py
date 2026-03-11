from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import io
from PIL import Image
from inference.inference_engine import DehazeInference

app = FastAPI(title="AtmosDehaze AI API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHECKPOINT_BEST = "outputs/checkpoints/mamba_best.pth"
CHECKPOINT_LAST = "outputs/checkpoints/mamba_last.pth"
TEMP_INPUT = "outputs/inference/temp_input.png"
TEMP_OUTPUT = "outputs/inference/latest_dehazed.png"

engine = None

def get_engine():
    global engine
    checkpoint = CHECKPOINT_BEST if os.path.exists(CHECKPOINT_BEST) else CHECKPOINT_LAST
    if engine is None and os.path.exists(checkpoint):
        try:
            engine = DehazeInference(checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    return engine

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Accept any image type that PIL can handle
    if not (image.content_type.startswith("image/") or image.filename.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.webp', '.avif', '.bmp', '.tiff', '.tif')
    )):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {image.content_type}")
    
    # Ensure inference directory exists
    os.makedirs(os.path.dirname(TEMP_INPUT), exist_ok=True)
    
    # Save uploaded file
    contents = await image.read()
    with open(TEMP_INPUT, "wb") as f:
        f.write(contents)
    
    infra_engine = get_engine()
    if not infra_engine:
        raise HTTPException(status_code=404, detail="Model checkpoint not found. Please train the model first.")
    
    try:
        # Run inference - returns (Final, Physics, Refine, Metadata)
        dehazed, physics, refine, metadata = infra_engine.predict(TEMP_INPUT)
        
        # Helper to convert PIL to Base64
        import base64
        def to_b64(img):
            b = io.BytesIO()
            img.save(b, format='PNG')
            return base64.b64encode(b.getvalue()).decode('utf-8')

        return JSONResponse({
            "success": True,
            "final": to_b64(dehazed),
            "physics": to_b64(physics),
            "refinement": to_b64(refine),
            "metadata": metadata
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/status")
async def status():
    checkpoint = CHECKPOINT_BEST if os.path.exists(CHECKPOINT_BEST) else CHECKPOINT_LAST
    return JSONResponse({
        "model_loaded": get_engine() is not None,
        "checkpoint": os.path.basename(checkpoint) if os.path.exists(checkpoint) else "None",
        "device": "NVIDIA RTX A6000" if get_engine() else "N/A"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
