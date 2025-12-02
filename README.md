# Text Summarizer using BART (Fine-Tuned on SAMSum)




This project implements an end-to-end dialogue summarization system using a fine-tuned **BART Large CNN** model on the **SAMSum dataset**. It includes a complete machine learning pipeline for training, evaluation, and a **FastAPI-based** inference service.

## Features

- **Fully automated training pipeline** (Data Ingestion -> Validation -> Transformation -> Training -> Evaluation)
- **Fine-tuned Model:** Uses `facebook/bart-large-cnn` optimized for conversational text.
- **ROUGE-based Evaluation:** Comprehensive metric tracking.
- **REST API:** Fast real-time summarization using FastAPI.
- **Batch Prediction:** Support for generating summaries from CSV files.



## Installation

1. **Create a Conda environment:**
   ```bash
   conda create -n texts python=3.10 -y
   conda activate texts
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Training the Model

To run the full pipeline, execute the main script:

```bash
python main.py
```

**This command executes the following steps sequentially:**
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation

The fine-tuned model will be saved inside the `artifacts` directory.

---

## Model Evaluation Results (ROUGE Scores)

 The model achieved the following ROUGE scores on the SAMSum test set. These scores indicate strong summarisation quality for a BART model fine-tuned on conversational data.

| Metric | Score |
| :--- | :--- |
| **ROUGE-1** | 0.4283 |
| **ROUGE-2** | 0.2244 |
| **ROUGE-L** | 0.3180 |
| **ROUGE-Lsum** | 0.3202 |

---

## Running the FastAPI Server

### 1. Start the Inference API
```bash
python app.py
```

### 2. Access the Web UI
Open your browser and navigate to the Swagger UI documentation:
[http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)

### 3. API Usage Example

**Endpoint:** `/predict` (or via the UI)

**Request Body:**
```json
{
  "text": "Person A: Hey, did you finish the presentation for tomorrow’s meeting? Person B: Not yet. I have been dealing with some issues in the data pipeline, and it's taking longer than expected. Person A: Do you need any help? I can review the slides or clean up some sections. Person B: That would be great. If you can finalize the last two slides, I’ll finish the charts. Person A: Sure, just send me the draft. Let's try to wrap everything up before dinner so we’re not rushing at night. Person B: Agreed. I’ll send it in 10 minutes."
}
```

**Response:**
```json
{
  "summary": "Person B hasn't finished the presentation for tomorrow's meeting yet because of issues in the data pipeline. Person A will review the slides and clean up some sections. Person B will finish the last two slides and the charts. They will try to wrap everything up before dinner so they're not rushing at night."
}
```

---

## Batch Prediction (CSV Input)

To generate summaries for multiple entries in your dataset at once:

```bash
python prediction.py
```

This script reads your SAMSum test CSV and produces a new file: `samsum-test_with_predictions.csv`.

---

## Notebook Evaluation

The `texts.ipynb` notebook is provided for experimental analysis. It includes:
- Model loading
- Manual predictions
- Corrected ROUGE evaluation using the HuggingFace `evaluate` library
- Comparison against baseline BART performance

---

## Acknowledgements

- SAMSum Dataset
- HuggingFace Transformers
- ROUGE Metric (Evaluate Library)
- FastAPI
```
