from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from src.pipeline.prediction import PredictionPipelines
from extractor import extract_text_from_docx, extract_text_from_pdf



app = FastAPI(title="Fake News Detection API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pipeline_instance=PredictionPipelines()
@app.get("/")
async def root():
    return {"message": "✅ Fake News Detector API is running"}



@app.post("/predict_files")
async def predict_from_files(file: UploadFile = File(...)):
    if pipeline_instance is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        
        if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
            raise HTTPException(status_code=422, detail="Unsupported File Format. Upload .docx or .pdf")

       
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file.file)
        else:
            text = extract_text_from_docx(file.file)

       
        if not text or not text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from file")

    
        resp = pipeline_instance.get_output(text)

        return JSONResponse(
            status_code=200,
            content={
                "label": resp.get("label"),
                "probability": resp.get("probability")
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"⚠️ Error in predict_from_files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



@app.post("/predict_from_text")
async def predict_from_text(text: str):
    if pipeline_instance is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        resp = pipeline_instance.get_output(text)
        return JSONResponse(
            status_code=200,
            content={
                "label": resp.get("label"),
                "probability": resp.get("probability")
            }
        )
    except Exception as e:
        print(f"⚠️ Error in predict_from_text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

