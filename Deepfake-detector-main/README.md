# Deepfake Detector

Streamlit-based image/audio deepfake screening application using Hugging Face classifiers, LangGraph orchestration, and optional Groq explanations.

## Detection improvements

The inference pipeline now uses a conservative decision policy instead of blindly accepting the top model label:

- consistent RGB + 224x224 image preprocessing before image inference
- explicit REAL and DEEPFAKE probability aggregation
- verified label mapping; unknown model labels are not silently treated as REAL
- confidence and decision-margin checks
- image quality gating for very small, extremely dark/bright, or heavily blurred images
- `UNKNOWN` / `UNCERTAIN` result when evidence is insufficient
- real/deepfake probabilities and decision margin shown in the UI and reports
- Groq explanations are explicitly treated as explanations of a model prediction, not forensic proof

## Important limitation

These changes make the application more conservative and reduce overconfident false positives, but they do **not** prove universal accuracy. A deepfake detector must be evaluated on a representative, unseen labeled dataset before a threshold can be claimed as calibrated.

## Evaluate on labeled test data

Create:

```text
dataset/
  real/
    image1.jpg
    image2.png
  deepfake/
    fake1.jpg
    fake2.png
```

Then run:

```bash
python evaluate_detector.py --data dataset
```

The evaluator reports accuracy, precision, recall, F1, ROC-AUC, false-positive rate, false-negative rate, confusion matrix, and the number of skipped/uncertain images. It writes `evaluation_results.json` and never fabricates metrics.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `GROQ_API_KEY` only if you want LLM-generated explanations. The detector itself does not require Groq.
