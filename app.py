from fastapi import FastAPI
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn
import os

from textSummarizer.pipeline.prediction import PredictionPipeline


# ---------- Request schema ----------

class TextRequest(BaseModel):
    text: str


# ---------- App & global objects ----------

app = FastAPI(
    title="Text Summarizer API",
    description="API for training and summarization using your fine-tuned model",
    version="1.0.0",
)

# Load the model once when the app starts
prediction_pipeline = PredictionPipeline()


# ---------- Routes ----------

@app.get("/", tags=["root"])
async def index():
    # Redirect to Swagger docs
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["train"])
async def training():
    """
    Trigger your training pipeline by calling main.py
    """
    try:
        exit_code = os.system("python main.py")
        if exit_code != 0:
            return Response(
                content=f"Training failed with exit code {exit_code}",
                media_type="text/plain",
                status_code=500,
            )
        return Response("Training successful !!", media_type="text/plain")
    except Exception as e:
        return Response(
            content=f"Error Occurred! {e}",
            media_type="text/plain",
            status_code=500,
        )


@app.post("/predict", tags=["prediction"])
async def predict_route(request: TextRequest):
    """
    Get a summary for the provided text.
    Request body: { "text": "your long dialogue/article here" }
    """
    try:
        summary = prediction_pipeline.predict(request.text)
        return {"summary": summary}
    except Exception as e:
        # FastAPI will convert this into a proper 500 response
        raise e


# ---------- Entry point (for running directly) ----------

if __name__ == "__main__":
    # Run with: python app.py
    uvicorn.run(app, host="0.0.0.0", port=8080)
