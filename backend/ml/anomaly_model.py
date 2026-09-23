from pathlib import Path

from sklearn.ensemble import IsolationForest
import joblib
import numpy as np

from .vectorizer import load_vectorizer, fit_vectorizer, transform

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'
DEFAULT_MODEL_PATH = MODEL_DIR / 'anomaly_model.pkl'


def create_model(contamination: float = 0.15, random_state: int = 42) -> IsolationForest:
    return IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=150,
        max_samples='auto',
    )


def train(
    messages: list[str],
    model_path: str | Path | None = None,
    vectorizer_path: str | Path | None = None,
) -> tuple[IsolationForest, "TfidfVectorizer"]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer = fit_vectorizer(messages, vectorizer_path)
    X = vectorizer.transform(messages)
    model = create_model()
    model.fit(X)
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    joblib.dump(model, path)
    return model, vectorizer


def load_model(path: str | Path | None = None) -> IsolationForest:
    p = Path(path) if path else DEFAULT_MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(f"Model not found at {p}. Run training first.")
    return joblib.load(p)


def predict(
    messages: list[str],
    model: IsolationForest | None = None,
    vectorizer=None,
) -> list[dict]:
    if model is None:
        model = load_model()
    if vectorizer is None:
        vectorizer = load_vectorizer()
    X = vectorizer.transform(messages)
    preds = model.predict(X)  
    scores = model.decision_function(X)  

    
    empty_rows = (X.sum(axis=1) == 0).A1  

    
    
    
    
    
    
    
    
    
    
    
    score_min, score_max = -0.3, 0.2  
    normalized_scores = (scores - score_min) / (score_max - score_min)
    normalized_scores = np.clip(normalized_scores, 0, 1)
    
    
    
    confidences = np.where(
        preds == -1,  
        1 - normalized_scores,  
        normalized_scores * 0.6 + 0.1  
    )
    
    
    scale_factor = 4.0  
    confidences = 1 / (1 + np.exp(-scale_factor * (confidences - 0.5)))
    
    
    confidences = np.clip(confidences, 0.1, 0.9)

    
    STRONG_ANOMALY_KEYWORDS = {
        'failed', 'failure', 'timeout', 'error', 'exception', 'critical',
        'fatal', 'refused', 'unauthorized', 'forbidden', 'injection',
        'attack', 'breach', 'overflow', 'crash', 'panic', 'abort'
    }
    
    results = []
    for idx, (pred, score, conf) in enumerate(zip(preds, scores, confidences)):
        is_anomaly = pred == -1
        message_lower = messages[idx].lower()
        
        
        if empty_rows[idx]:
            is_anomaly = True
            conf = max(conf, 0.92)
        
        elif any(keyword in message_lower for keyword in STRONG_ANOMALY_KEYWORDS):
            if not is_anomaly:  
                is_anomaly = True
            
            keyword_count = sum(1 for keyword in STRONG_ANOMALY_KEYWORDS if keyword in message_lower)
            boost = min(0.15 + (keyword_count * 0.1), 0.4)  
            conf = min(max(conf, 0.6) + boost, 0.95)  
        
        results.append({
            'prediction': 'Anomaly' if is_anomaly else 'Normal',
            'confidence': round(float(conf), 2),
        })
    return results


def predict_single(message: str, model=None, vectorizer=None) -> dict:
    return predict([message], model, vectorizer)[0]


if __name__ == '__main__':
    from backend.parser import parse_logs
    from backend.preprocessing import process_logs

    with open('datasets/log_example.txt', 'r') as f:
        content = f.read()
    parsed = parse_logs(content)
    processed = process_logs(parsed)
    messages = [p.get('cleaned_message') or p.get('message', '') for p in processed]
    messages = [m for m in messages if m]
    if messages:
        train(messages)
        print("Trained. Sample predict:", predict_single(messages[0]))
