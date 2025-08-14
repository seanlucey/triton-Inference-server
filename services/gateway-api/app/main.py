from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .triton_client import Triton

app = FastAPI(title="Gateway API", version="0.1.0")
triton = Triton()

@app.get("/healthz")
def healthz():
    # Optionally call triton /v2/health/ready
    return {"status": "ok"}

@app.post("/infer")
async def infer(file: UploadFile = File(...), topk: int = 5):
    blob = await file.read()
    try:
        indices, scores = triton.run_ensemble(blob, topk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Convert to JSON; label mapping can be added here
    return JSONResponse({
        "topk": topk,
        "indices": indices.tolist(),
        "scores": scores.tolist(),
    })
