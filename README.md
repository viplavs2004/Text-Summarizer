Text Summarizer using BART (Fine-Tuned on SAMSum)

This project implements an end-to-end dialogue summarization system using a fine-tuned BART Large CNN model on the SAMSum dataset.
It includes a complete machine learning pipeline for training, evaluation, and a FastAPI-based inference service.

Features

Fully automated training pipeline (data ingestion → training → evaluation)

Fine-tuned facebook/bart-large-cnn model

Summarization optimized for conversational text

ROUGE-based evaluation

REST API for real-time summarization

Batch prediction support for CSV files

Installation
1. Create Conda environment
'''conda create -n texts python=3.10 -y
conda activate texts'''

2. Install dependencies
''' pip install -r requirements.txt '''

Training the Model

Run the full ML pipeline:

python main.py


This executes:

Data ingestion

Data validation

Data transformation

Model training

Model evaluation

The fine-tuned model is saved inside the artifacts directory.

Model Evaluation Results (ROUGE Scores)

After correcting the evaluation script, the model achieved the following ROUGE scores on the SAMSum test set:

Metric	Score
ROUGE-1	0.4283
ROUGE-2	0.2244
ROUGE-L	0.3180
ROUGE-Lsum	0.3202

These scores indicate strong summarization quality for a BART model fine-tuned on conversational data.

Running the FastAPI Server

Start the inference API:

python app.py


Open the web UI:

http://127.0.0.1:8080/docs

Example request
{
  "text": "Person A: Hey, did you finish the presentation for tomorrow’s meeting? Person B: Not yet. I have been dealing with some issues in the data pipeline, and it's taking longer than expected. Person A: Do you need any help? I can review the slides or clean up some sections. Person B: That would be great. If you can finalize the last two slides, I’ll finish the charts. Person A: Sure, just send me the draft. Let's try to wrap everything up before dinner so we’re not rushing at night. Person B: Agreed. I’ll send it in 10 minutes."
}

Example response
{
  "summary": "Person B hasn't finished the presentation for tomorrow's meeting yet because of issues in the data pipeline. Person A will review the slides and clean up some sections. Person B will finish the last two slides and the charts. They will try to wrap everything up before dinner so they're not rushing at night."
}

Batch Prediction (CSV Input)

Generate summaries for multiple entries in your dataset:

python prediction.py


This reads your SAMSum test CSV and produces:

samsum-test_with_predictions.csv

Notebook Evaluation

The texts.ipynb notebook includes:

Model loading

Manual predictions

Corrected ROUGE evaluation using HuggingFace evaluate

Comparison against baseline BART performance

Acknowledgements

SAMSum dataset

HuggingFace Transformers

ROUGE metric (evaluate library)

FastAPI for deployment