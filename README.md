# live-sentiment-analyzer

A real-time sentiment analysis system built with **DistilBERT**, fine-tuned locally on the SST-2 dataset. Given a text input, the model classifies it as **POSITIVE** or **NEGATIVE** with a confidence score.

> **Live Demo**: [HuggingFace Spaces](https://huggingface.co/spaces/mat-rib/live-sentiment-analyzer)

> **Note on confidence score**: the score does not indicate *how positive or negative* a text is — it represents *how confident the model is* about its prediction. A score of 0.99 on a NEGATIVE label means the model is 99% sure the text is negative, not that it is "99% negative".

---

## Overview

This project covers the full ML pipeline — from fine-tuning a transformer model to serving it via a REST API and exposing it through an interactive interface.

| Component | Technology |
|---|---|
| Model | DistilBERT (fine-tuned on SST-2) |
| Backend API | FastAPI |
| Local Interface | Streamlit |
| Demo Interface | Gradio (HuggingFace Spaces) |
| Model Hosting | HuggingFace Model Hub |

> **Note on interfaces**: the local version uses **Streamlit** + a separate **FastAPI** backend. The hosted demo uses **Gradio**, as required by HuggingFace Spaces. The model and prediction logic are identical — only the interface layer differs.


The model uses **transfer learning**: DistilBERT is pre-trained on a large corpus and fine-tuned on the SST-2 sentiment dataset. Only 2,000 training examples were used to keep training time low, achieving ~85% accuracy on the validation set.

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/mat-riboni/live-sentiment-analyzer
cd live-sentiment-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python model/train.py
```

This will fine-tune DistilBERT on SST-2 and save the model to `model/fine_tuned/`.

> The script uses MPS (Apple Silicon), CUDA (NVIDIA), or CPU — automatically detected.

### 5. Start the API

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.  
Interactive docs at `http://127.0.0.1:8000/docs`.

### 6. Start the Streamlit interface

Open a second terminal (keep the API running) and launch:

```bash
streamlit run app/app.py
```

The interface will open automatically at `http://localhost:8501`.

---

## API Reference

### `POST /predict`

Classifies a single text input.

**Request**
```json
{ "text": "This movie was absolutely amazing!" }
```

**Response**
```json
{
  "label": "POSITIVE",
  "score": 0.9987,
  "text": "This movie was absolutely amazing!"
}
```

---

## Model Performance

| Metric | Value |
|---|---|
| Accuracy | 85.75% |
| F1 Score | 0.8576 |
| Training examples | 2,000 |
| Validation examples | 400 |
| Epochs | 10 |

> Training was intentionally limited to 2,000 examples for speed. Training on the full SST-2 dataset (~67,000 examples) with 2–3 epochs would significantly reduce overfitting and improve generalization.

---

## Requirements

```
transformers
datasets
torch
scikit-learn
fastapi
uvicorn
streamlit
accelerate>=1.1.0
```
