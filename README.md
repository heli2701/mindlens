# 🧠 MindLens

A mental health text classifier that reads Reddit-style posts and predicts one of 5 categories:
**Depression · Anxiety · Suicidal Ideation · Stress · Normal**

Built on `mental/mental-roberta-base` fine-tuned on 50,000 real Reddit posts.

## Demo
> Try the live API: [your deployed URL here]

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | XX% |
| Macro F1 | X.XX |

## Project Structure
app.py                          ← Flask REST API
index.html                      ← Frontend UI
MindLens_Mac_M4_commented.ipynb ← Training notebook
mindlens_model/label_map.json   ← Class labels config
requirements.txt                ← Python dependencies
## Model Weights
Weights are hosted on HuggingFace (too large for GitHub):
👉 https://huggingface.co/YOUR_USERNAME/mindlens-weights

## How to Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Then open http://localhost:10000

## How It Works
1. Text is tokenised with `mental/mental-roberta-base`
2. Auxiliary features (sentiment, length, crisis keywords) are computed
3. A classification head outputs probabilities for 5 classes
4. SHAP explains which words drove the prediction