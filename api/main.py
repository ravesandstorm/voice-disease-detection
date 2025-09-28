from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

# Import functions from service.py
import service

app = FastAPI(title="Voice Disease Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        return await service.predict_audio(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/random-samples/")
async def get_random_samples():
    try:
        return service.get_random_samples()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def get_index():
    return FileResponse("../frontend/index.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)