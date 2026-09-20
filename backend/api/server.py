"""
FastAPI backend for Log Anomaly Detection.
Endpoints: POST /analyze-log, GET /logs, GET /stats, GET /recent-anomalies
"""
import sys
from pathlib import Path

# Ensure backend is on path (backend_root = backend folder)
backend_root = Path(__file__).resolve().parent.parent
project_root = backend_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Backend imports (run from project root: uvicorn backend.api.server:app)
from backend.parser.log_parser import parse_logs
from backend.preprocessing.log_processor import process_logs, clean_message
from backend.ml.anomaly_model import predict
from backend.rules.rule_engine import run_rules_single, combine_with_ml

app = FastAPI(title="Log Anomaly Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for processed logs (for GET /logs, /stats, /recent-anomalies)
_stored_logs: list[dict] = []
_stored_predictions: list[dict] = []


# Attempt to ensure an ML model is present when the app starts. If the
# serialized model/vectorizer files are missing or broken we fall back to
# training on whatever dataset files are available. This prevents the API
# from returning all-"Normal" predictions simply because there is no model.
@app.on_event("startup")
async def ensure_ml_model():
    from backend.train import load_messages
    from backend.ml.anomaly_model import train as train_model
    # load existing components; _get_ml_components returns (None,None) if missing
    model, vectorizer = _get_ml_components()
    if model is None or vectorizer is None:
        msgs = load_messages()
        if msgs:
            train_model(msgs)
            print(f"[startup] auto-trained ML model on {len(msgs)} messages")
        else:
            print("[startup] no training messages found; ML backend will use rule-only fallback")


class AnalyzeRequest(BaseModel):
    message: str


class AnalyzeResponse(BaseModel):
    message: str
    prediction: str
    confidence: float
    source: str


def _get_ml_components():
    """Lazy load model and vectorizer."""
    model_path = backend_root / "models" / "anomaly_model.pkl"
    vec_path = backend_root / "models" / "tfidf_vectorizer.pkl"
    if not model_path.exists():
        return None, None
    try:
        import joblib
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        return model, vectorizer
    except Exception:
        return None, None


@app.post("/analyze-log", response_model=AnalyzeResponse)
async def analyze_log(req: AnalyzeRequest):
    """Analyze a single log message and return prediction."""
    model, vectorizer = _get_ml_components()
    if model is None or vectorizer is None:
        raise HTTPException(500, "ML model not trained. Run: python backend/train.py")
    # Preprocess
    cleaned = clean_message(req.message)
    if not cleaned:
        cleaned = req.message[:500]
    # ML prediction
    ml_results = predict([cleaned], model, vectorizer)
    ml_pred = ml_results[0]
    # Rule check (single log, no context)
    parsed = parse_logs(req.message)
    log_dict = parsed[0] if parsed else {"message": req.message, "raw": req.message}
    rule_result = run_rules_single(log_dict, [log_dict], 0)
    final_pred, source, reason = combine_with_ml(rule_result, ml_pred["prediction"])
    confidence = ml_pred["confidence"]
    return AnalyzeResponse(
        message=req.message[:500],
        prediction=final_pred,
        confidence=confidence,
        source=source,
    )


@app.get("/logs")
async def get_logs():
    """Return processed logs with predictions and detection reasons."""
    logs_with_pred = []
    for log, pred in zip(_stored_logs, _stored_predictions):
        logs_with_pred.append({
            **log,
            "prediction": pred.get("prediction", "Normal"),
            "confidence": pred.get("confidence", 0),
            "source": pred.get("source", "ML"),
            "detection_reason": pred.get("detection_reason", ""),
        })
    return {"logs": logs_with_pred}


@app.get("/stats")
async def get_stats():
    """Dashboard statistics."""
    total = len(_stored_logs)
    anomalies = sum(1 for p in _stored_predictions if p.get("prediction") == "Anomaly")
    normal = total - anomalies
    pct = round(100.0 * anomalies / total, 1) if total else 0
    return {
        "total_logs": total,
        "anomaly_count": anomalies,
        "normal_count": normal,
        "anomaly_percentage": pct,
    }


@app.get("/features")
async def get_features():
    """Return extracted features for dashboard feature table."""
    try:
        # Import the advanced feature extractor
        from backend.ml.advanced_features import AdvancedFeatureExtractor
        
        # Initialize feature extractor
        extractor = AdvancedFeatureExtractor(window_size_minutes=10)
        
        # Extract features from stored logs
        if not _stored_logs:
            return {"features": []}
        
        # Convert logs to format expected by feature extractor
        log_messages = [log.get("message", log.get("raw", "")) for log in _stored_logs]
        
        # Extract features
        features = extractor.extract_features_from_logs(log_messages)
        
        # Convert to FeatureVector format
        feature_vectors = []
        for i, feature in enumerate(features):
            feature_vector = {
                "timeWindow": f"Window {i+1}",
                "errorCount": feature.get("error_count", 0),
                "warnCount": feature.get("warn_count", 0), 
                "uniqueTemplates": feature.get("unique_templates", 1),
                "avgResponseTime": feature.get("avg_response_time", 200),
                "eventFrequency": feature.get("event_frequency", 1.0),
                "stdDeviation": feature.get("std_deviation", 0.0)
            }
            feature_vectors.append(feature_vector)
        
        return {"features": feature_vectors[-50:]}  # Return last 50 windows
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        # Fallback: return basic computed features
        feature_vectors = []
        from collections import defaultdict
        time_windows = defaultdict(list)
        
        # Group logs by time windows
        for log in _stored_logs:
            timestamp = log.get("timestamp", "")
            if timestamp:
                time_window = timestamp[:16]  # YYYY-MM-DD HH:MM
                time_windows[time_window].append(log)
        
        # Calculate basic features
        for time_window, entries in sorted(time_windows.items()):
            error_count = sum(1 for e in entries if e.get("log_level") == "ERROR")
            warn_count = sum(1 for e in entries if e.get("log_level") == "WARN")
            
            feature_vector = {
                "timeWindow": time_window,
                "errorCount": error_count,
                "warnCount": warn_count,
                "uniqueTemplates": len(set(e.get("template", "") for e in entries if e.get("template"))),
                "avgResponseTime": 200 + (error_count * 100),  # Simulated response time
                "eventFrequency": len(entries) / 10.0,  # Events per minute
                "stdDeviation": min(error_count * 15.5, 50.0)  # Simulated standard deviation
            }
            feature_vectors.append(feature_vector)
        
        return {"features": feature_vectors[-50:]}


@app.get("/recent-anomalies")
async def get_recent_anomalies():
    """Return last detected anomalies with metrics for dashboard charts."""
    # Generate time series data with metrics
    anomaly_data = []
    
    # Group logs by time windows (e.g., every minute or hour)
    from collections import defaultdict
    time_windows = defaultdict(list)
    
    for log, pred in zip(_stored_logs, _stored_predictions):
        timestamp = log.get("timestamp", "")
        if timestamp:
            # Simple time window grouping (by minute)
            time_window = timestamp[:16]  # YYYY-MM-DD HH:MM
            time_windows[time_window].append({
                "log": log,
                "prediction": pred,
                "is_anomaly": pred.get("prediction") == "Anomaly"
            })
    
    # Calculate metrics for each time window
    for time_window, entries in sorted(time_windows.items()):
        anomaly_entries = [e for e in entries if e["is_anomaly"]]
        total_entries = len(entries)
        
        # Calculate metrics
        error_count = sum(1 for e in entries if e["log"].get("log_level") == "ERROR")
        warn_count = sum(1 for e in entries if e["log"].get("log_level") == "WARN")
        
        # Simulate response time based on log patterns
        avg_response_time = 200 + (len(anomaly_entries) * 50) + (error_count * 100)
        
        # Calculate anomaly score based on confidence and count
        anomaly_score = 0.0
        if anomaly_entries:
            avg_confidence = sum(e["prediction"].get("confidence", 0) for e in anomaly_entries) / len(anomaly_entries)
            anomaly_score = min(avg_confidence + (len(anomaly_entries) / total_entries) * 0.3, 1.0)
        
        # Get the most recent anomaly entry for details
        latest_anomaly = None
        if anomaly_entries:
            latest_anomaly = max(anomaly_entries, key=lambda x: x["log"].get("timestamp", ""))
        
        # Create RecentAnomaly format with embedded metrics
        if latest_anomaly or error_count > 0:
            anomaly_entry = {
                "message": (latest_anomaly["log"].get("message", "") if latest_anomaly else f"{error_count} errors detected")[:300],
                "prediction": "Anomaly" if (latest_anomaly or error_count > 0) else "Normal",
                "confidence": anomaly_score if anomaly_score > 0 else (latest_anomaly["prediction"].get("confidence", 0.5) if latest_anomaly else 0.5),
                "source": latest_anomaly["prediction"].get("source", "ML") if latest_anomaly else "Statistical",
                "timestamp": time_window,
                "detection_reason": latest_anomaly["prediction"].get("detection_reason", "") if latest_anomaly else f"High error count ({error_count})",
                "service": latest_anomaly["log"].get("service", "System") if latest_anomaly else "System",
                "log_level": "ERROR" if error_count > 0 else (latest_anomaly["log"].get("log_level", "INFO") if latest_anomaly else "INFO"),
                # Additional metrics for charts (embedded in the response)
                "errorCount": error_count,
                "avgResponseTime": avg_response_time,
                "uniqueTemplates": len(set(e["log"].get("template", "") for e in entries if e["log"].get("template"))),
                "anomalyScore": anomaly_score,
                "isAnomaly": True,
                "hybridDecision": "anomaly"
            }
            anomaly_data.append(anomaly_entry)
    
    return {"anomalies": anomaly_data[-50:]}


class IngestRequest(BaseModel):
    content: str


class DatasetSplitRequest(BaseModel):
    test_ratio: float = 0.2
    random_state: int = 42


class DatasetAnalyzeRequest(BaseModel):
    paths: list[str] = []


@app.get("/datasets")
async def get_datasets():
    """List recursively discovered source datasets and generated split status."""
    from backend.train import DATASET_DIR, TEST_DATASET_DIR, TRAIN_DATASET_DIR, discover_dataset_paths

    datasets = []
    for path in discover_dataset_paths(TEST_DATASET_DIR):
        try:
            parsed = parse_logs(path.read_text(encoding="utf-8", errors="ignore"))
            records = len(parsed)
        except OSError:
            records = 0
        datasets.append({
            "path": str(path.relative_to(TEST_DATASET_DIR)),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "records": records,
        })
    return {
        "datasets": datasets,
        "split": {
            "available": TEST_DATASET_DIR.exists() and bool(discover_dataset_paths(TEST_DATASET_DIR)),
            "train_path": str(TRAIN_DATASET_DIR.relative_to(DATASET_DIR)),
            "test_path": str(TEST_DATASET_DIR.relative_to(DATASET_DIR)),
        },
    }


@app.post("/datasets/split")
async def split_datasets(req: DatasetSplitRequest):
    """Create deterministic recursive train/test files from source datasets."""
    from backend.train import split_dataset

    try:
        return {"message": "Dataset split created", "split": split_dataset(req.test_ratio, req.random_state)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/datasets/train")
async def train_split_datasets():
    """Train the saved model using datasets/train only."""
    from backend.train import TEST_DATASET_DIR, TRAIN_DATASET_DIR, load_processed_logs, split_dataset
    from backend.ml.anomaly_model import train as train_model

    if not TRAIN_DATASET_DIR.exists():
        raise HTTPException(status_code=400, detail="No datasets/train directory found")
    if not TEST_DATASET_DIR.exists() or not any(TEST_DATASET_DIR.rglob("*")):
        try:
            split_dataset()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    records, sources = load_processed_logs(TRAIN_DATASET_DIR)
    messages = [record.get("cleaned_message") or record.get("message", "") for record in records]
    if not messages:
        raise HTTPException(status_code=400, detail="datasets/train contains no parseable logs")
    train_model(messages)
    return {"message": "Model trained on generated training split", "train_sources": sources, "train_records": len(messages)}


@app.post("/datasets/evaluate")
async def evaluate_split_datasets():
    """Evaluate the saved model against generated test records only."""
    from backend.train import TEST_DATASET_DIR, TRAIN_DATASET_DIR, load_processed_logs
    from backend.ml.evaluation import build_hybrid_predictions, build_silver_labels, evaluate_binary

    if not TRAIN_DATASET_DIR.exists() or not TEST_DATASET_DIR.exists():
        raise HTTPException(status_code=400, detail="Create datasets/test by training first")
    model, vectorizer = _get_ml_components()
    if model is None or vectorizer is None:
        raise HTTPException(status_code=400, detail="Train the model before evaluating the test split")

    test_logs, sources = load_processed_logs(TEST_DATASET_DIR)
    if not test_logs:
        raise HTTPException(status_code=400, detail="The generated test split contains no parseable logs")
    test_messages = [log.get("cleaned_message") or log.get("message", "") for log in test_logs]
    labels = build_silver_labels(test_logs)
    ml_output = predict(test_messages, model, vectorizer)
    ml_labels = [1 if result["prediction"] == "Anomaly" else 0 for result in ml_output]
    hybrid_labels = build_hybrid_predictions(test_logs, ml_labels)
    return {
        "test_sources": sources,
        "test_records": len(test_logs),
        "labeling": "silver",
        "ml_only": evaluate_binary(labels, ml_labels),
        "hybrid_ml_rules": evaluate_binary(labels, hybrid_labels),
    }


@app.post("/datasets/analyze")
async def analyze_datasets(req: DatasetAnalyzeRequest):
    """Analyze selected generated test files and publish them to the dashboard."""
    from backend.train import TEST_DATASET_DIR, discover_dataset_paths

    model, vectorizer = _get_ml_components()
    if model is None or vectorizer is None:
        raise HTTPException(status_code=400, detail="Train the model before analyzing test datasets")

    available = [path.resolve() for path in discover_dataset_paths(TEST_DATASET_DIR)]
    selected = req.paths or [str(path.relative_to(TEST_DATASET_DIR)) for path in available]
    selected_paths = []
    test_root = TEST_DATASET_DIR.resolve()
    for relative_path in selected:
        candidate = (TEST_DATASET_DIR / relative_path).resolve()
        try:
            candidate.relative_to(test_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid test dataset path: {relative_path}") from exc
        if candidate not in available:
            raise HTTPException(status_code=404, detail=f"Test dataset not found: {relative_path}")
        selected_paths.append(candidate)

    parsed_logs = []
    for path in selected_paths:
        parsed_logs.extend(parse_logs(path.read_text(encoding="utf-8", errors="ignore")))
    processed = process_logs(parsed_logs)
    messages = [log.get("cleaned_message") or log.get("message", "") for log in processed]
    ml_predictions = predict(messages, model, vectorizer)

    results = []
    for index, (log, prediction) in enumerate(zip(processed, ml_predictions)):
        rule_result = run_rules_single(log, processed, index)
        dataset_label = log.get("dataset_label")
        if dataset_label in (0, 1):
            final = "Anomaly" if dataset_label else "Normal"
            source = "Dataset label"
            reason = "BGL benchmark label"
        else:
            final, source, reason = combine_with_ml(rule_result, prediction["prediction"])
        results.append({
            "prediction": final,
            "confidence": prediction["confidence"],
            "source": source,
            "detection_reason": reason,
            "rule_triggered": rule_result.triggered if rule_result else False,
        })

    global _stored_logs, _stored_predictions
    _stored_logs = processed
    _stored_predictions = results
    return {
        "message": "Selected test datasets analyzed",
        "paths": selected,
        "count": len(processed),
        "anomalies": sum(1 for result in results if result["prediction"] == "Anomaly"),
    }


@app.post("/ingest")
async def ingest_logs(req: IngestRequest):
    """Ingest and process a batch of logs. Used by upload flow."""
    content = req.content
    global _stored_logs, _stored_predictions
    parsed = parse_logs(content)
    if not parsed:
        return {"message": "No parseable logs", "count": 0}
    processed = process_logs(parsed)
    messages = [p.get("cleaned_message") or p.get("message", "") for p in processed]

    model, vectorizer = _get_ml_components()
    if model is None or vectorizer is None:
        # Attempt on-the-fly training using available datasets so that incoming
        # ingest requests still benefit from ML. This is slow but better than
        # failing silently with all-Normal results.
        try:
            from backend.train import load_messages, train as train_model
            msgs = load_messages()
            if msgs:
                train_model(msgs)
                model, vectorizer = _get_ml_components()
        except Exception:
            pass

    if model is None or vectorizer is None:
        # Fallback: mark all as Normal if still no model
        ml_preds = [{"prediction": "Normal", "confidence": 0.5} for _ in messages]
    else:
        ml_preds = predict(messages, model, vectorizer)
    results = []
    for i, (log, mp) in enumerate(zip(processed, ml_preds)):
        rule_res = run_rules_single(log, processed, i)
        dataset_label = log.get("dataset_label")
        if dataset_label in (0, 1):
            final = "Anomaly" if dataset_label else "Normal"
            source = "Dataset label"
            reason = "BGL benchmark label"
        else:
            final, source, reason = combine_with_ml(rule_res, mp["prediction"])
        results.append({
            "prediction": final,
            "confidence": mp["confidence"],
            "source": source,
            "detection_reason": reason,
            "rule_triggered": rule_res.triggered if rule_res else False,
        })
    _stored_logs = processed
    _stored_predictions = results
    return {"message": "OK", "count": len(processed), "anomalies": sum(1 for r in results if r["prediction"] == "Anomaly")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
