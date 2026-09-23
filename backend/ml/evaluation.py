from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from backend.rules.rule_engine import combine_with_ml, run_rules_single
from backend.ml.anomaly_model import predict
from backend.preprocessing.log_processor import clean_message


ERROR_LEVELS = {"ERROR", "CRITICAL", "FATAL"}
ANOMALY_KEYWORDS = (
    "failed",
    "failure",
    "timeout",
    "unauthorized",
    "forbidden",
    "exception",
    "critical",
    "attack",
    "injection",
    "refused",
    "overload",
)
ANOMALY_PATTERNS = [
    re.compile(r"\bauthentication\s+failed\b", re.I),
    re.compile(r"\bconnection\s+refused\b", re.I),
    re.compile(r"\bsql\s+injection\b", re.I),
    re.compile(r"\bheartbeat\s+failure\b", re.I),
]


def build_silver_labels(logs: list[dict]) -> list[int]:
    labels: list[int] = []
    for idx, log in enumerate(logs):
        dataset_label = log.get("dataset_label")
        if dataset_label in (0, 1):
            labels.append(int(dataset_label))
            continue
        msg = (log.get("message") or log.get("raw") or "").lower()
        level = str(log.get("log_level") or "").upper()
        rule_triggered = run_rules_single(log, logs, idx) is not None
        keyword_hit = any(k in msg for k in ANOMALY_KEYWORDS)
        pattern_hit = any(p.search(msg) for p in ANOMALY_PATTERNS)
        severe_level = level in ERROR_LEVELS
        labels.append(1 if (rule_triggered or severe_level or keyword_hit or pattern_hit) else 0)
    return labels


def evaluate_binary(y_true: list[int], y_pred: list[int]) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "labels": ["normal", "anomaly"],
            "matrix": cm,
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
        },
        "support": {
            "normal": int(sum(1 for v in y_true if v == 0)),
            "anomaly": int(sum(1 for v in y_true if v == 1)),
            "total": int(len(y_true)),
        },
    }


def build_hybrid_predictions(test_logs: list[dict], ml_labels: list[int]) -> list[int]:
    preds: list[int] = []
    for i, (log, ml_lab) in enumerate(zip(test_logs, ml_labels)):
        rule_res = run_rules_single(log, test_logs, i)
        ml_text = "Anomaly" if ml_lab == 1 else "Normal"
        final_text, _, _ = combine_with_ml(rule_res, ml_text)
        preds.append(1 if final_text == "Anomaly" else 0)
    return preds


def save_metrics(path: str | Path, payload: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _parse_label_value(v: object) -> int | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if int(v) in (0, 1):
            return int(v)
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    if s in {"0", "false", "normal"}:
        return 0
    if s in {"1", "true", "anomaly"}:
        return 1
    try:
        i = int(s)
        if i in (0, 1):
            return i
    except ValueError:
        return None
    return None


def load_gold_labels_csv(csv_path: str | Path) -> tuple[list[dict], list[str], list[int]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Gold labels CSV not found: {path}")

    from backend.parser.log_parser import parse_logs

    ml_messages: list[str] = []
    logs_for_rules: list[dict] = []
    y_true: list[int] = []

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return [], [], []

        fieldnames = {name.strip().lower(): name for name in reader.fieldnames}
        msg_field = (
            fieldnames.get("message")
            or fieldnames.get("raw_message")
            or fieldnames.get("raw")
            or fieldnames.get("log")
        )
        label_field = (
            fieldnames.get("label")
            or fieldnames.get("is_anomaly")
            or fieldnames.get("anomaly")
            or fieldnames.get("gold_label")
        )
        if not msg_field or not label_field:
            return [], [], []

        for row in reader:
            raw_msg = (row.get(msg_field) or "").strip()
            y = _parse_label_value(row.get(label_field))
            if not raw_msg or y is None:
                continue

            cleaned = clean_message(raw_msg)
            if not cleaned:
                cleaned = raw_msg[:500]

            parsed = parse_logs(raw_msg)
            log_dict = parsed[0] if parsed else {"message": raw_msg, "raw": raw_msg, "log_level": "INFO", "ip_address": None}
            logs_for_rules.append(log_dict)
            ml_messages.append(cleaned)
            y_true.append(y)

    return logs_for_rules, ml_messages, y_true


def evaluate_gold(
    gold_logs_for_rules: list[dict],
    gold_ml_messages: list[str],
    y_true: list[int],
    model,
    vectorizer,
) -> tuple[dict, dict]:
    ml_output = predict(gold_ml_messages, model, vectorizer)
    y_ml = [1 if o["prediction"] == "Anomaly" else 0 for o in ml_output]
    ml_metrics = evaluate_binary(y_true, y_ml)
    y_hybrid = build_hybrid_predictions(gold_logs_for_rules, y_ml)
    hybrid_metrics = evaluate_binary(y_true, y_hybrid)
    return ml_metrics, hybrid_metrics
