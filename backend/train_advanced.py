
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.parser.log_parser import parse_logs
from backend.preprocessing.log_processor import process_logs
from backend.rules.rule_engine import RuleEngine

from backend.ml.advanced_features import AdvancedFeatureExtractor
from backend.ml.advanced_vectorizer import AdvancedTextVectorizer
from backend.ml.advanced_detectors import AdvancedAnomalyDetector, HybridAnomalyDetector
from backend.ml.sequence_detectors import SequenceEnsembleDetector
from backend.ml.advanced_evaluation import AdvancedEvaluator, ModelComparisonReport

class AdvancedAnomalyPipeline:
    
    def __init__(self, 
                 contamination_rates: List[float] = [0.02, 0.05, 0.1],
                 sequence_length: int = 10,
                 batch_size: int = 1000):
        
        self.contamination_rates = contamination_rates
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        
        self.feature_extractor = AdvancedFeatureExtractor(
            window_size_minutes=5, 
            sequence_length=sequence_length
        )
        self.text_vectorizer = AdvancedTextVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            use_svd=True,
            svd_components=300
        )
        self.ml_detector = AdvancedAnomalyDetector(
            contamination_rates=contamination_rates
        )
        self.sequence_detector = SequenceEnsembleDetector(
            sequence_length=sequence_length
        )
        self.rule_engine = RuleEngine()
        self.evaluator = AdvancedEvaluator()
        
        self.training_data = {}
        self.model_results = {}
        self.comparison_results = {}
        
    def load_and_preprocess_data(self, dataset_paths: List[str]) -> Tuple[List[str], List[Dict]]:
        print("Loading and preprocessing data...")
        
        all_records = []
        all_sources = []
        
        for path in dataset_paths:
            print(f"  Loading: {path}")
            if os.path.exists(path):
                content = Path(path).read_text(encoding="utf-8", errors="ignore")
                parsed = parse_logs(content)
                processed = process_logs(parsed)
                valid = [p for p in processed if (p.get("cleaned_message") or p.get("message", ""))]
                all_records.extend(valid)
                all_sources.append(os.path.basename(path))
            else:
                print(f"  Warning: File not found - {path}")
        
        print(f"  Total records loaded: {len(all_records)}")
        
        messages = [(r.get("cleaned_message") or r.get("message", "")).strip() 
                   for r in all_records]
        messages = [m for m in messages if len(m) > 3]
        
        return messages, all_records
    
    def extract_features_batched(self, messages: List[str]) -> Tuple[np.ndarray, List[Dict]]:
        print(f"Extracting features from {len(messages)} messages...")
        
        all_features = []
        all_parsed = []
        
        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i:i + self.batch_size]
            print(f"  Processing batch {i//self.batch_size + 1}/{(len(messages)-1)//self.batch_size + 1}")
            
            batch_features, batch_parsed = self.feature_extractor.extract_all_features(batch_messages)
            all_features.append(batch_features)
            all_parsed.extend(batch_parsed)
        
        features = np.vstack(all_features)
        print(f"  Features extracted: {features.shape}")
        
        return features, all_parsed
    
    def vectorize_text(self, messages: List[str]) -> np.ndarray:
        print("Vectorizing text data...")
        
        text_features = self.text_vectorizer.fit_transform(messages)
        print(f"  Text features shape: {text_features.shape}")
        
        return text_features
    
    def create_labels(self, parsed_logs: List[Dict]) -> np.ndarray:
        print("Creating silver labels...")
        
        labels = []
        for log in parsed_logs:
            rule_triggers = self.rule_engine.evaluate_log(log)
            is_anomaly = len(rule_triggers) > 0
            
            message = log.get('message', '').lower()
            log_level = log.get('log_level', '')
            
            if log_level in ['ERROR', 'CRITICAL', 'FATAL']:
                is_anomaly = True
            elif 'failed' in message or 'timeout' in message or 'error' in message:
                is_anomaly = True
            
            labels.append(1 if is_anomaly else 0)
        
        labels_array = np.array(labels)
        print(f"  Labels created: {np.sum(labels_array)} anomalies out of {len(labels_array)}")
        
        return labels_array
    
    def train_models(self, features: np.ndarray, text_features: np.ndarray, 
                    messages: List[str], labels: np.ndarray) -> Dict[str, Any]:
        print("Training models...")
        
        combined_features = np.hstack([features, text_features])
        print(f"  Combined features shape: {combined_features.shape}")
        
        print("  Training ML anomaly detector...")
        self.ml_detector.fit(combined_features, labels)
        
        print("  Training sequence detector...")
        self.sequence_detector.fit(messages)
        
        print("  Computing rule-based scores...")
        rule_scores = self._compute_rule_scores(messages)
        
        self.training_data = {
            'features': features,
            'text_features': text_features,
            'combined_features': combined_features,
            'messages': messages,
            'labels': labels,
            'rule_scores': rule_scores
        }
        
        return self.training_data
    
    def _compute_rule_scores(self, messages: List[str]) -> np.ndarray:
        rule_scores = np.zeros(len(messages))
        
        for i, message in enumerate(messages):
            log_record = {'message': message, 'raw_message': message}
            
            rule_triggers = self.rule_engine.evaluate_log(log_record)
            
            score = len(rule_triggers) * 20
            rule_scores[i] = min(score, 100)
        
        return rule_scores
    
    def evaluate_all_models(self, test_split: float = 0.2) -> Dict[str, Any]:
        print("Evaluating models...")
        
        data = self.training_data
        X_combined = data['combined_features']
        y = data['labels']
        messages = data['messages']
        rule_scores = data['rule_scores']
        
        split_idx = int(len(X_combined) * (1 - test_split))
        
        X_train = X_combined[:split_idx]
        X_test = X_combined[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        messages_test = messages[split_idx:]
        rule_scores_test = rule_scores[split_idx:]
        
        print(f"  Train set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        print(f"  Test anomalies: {np.sum(y_test)} out of {len(y_test)}")
        
        print("  Evaluating ML models...")
        ml_evaluation = self.ml_detector.evaluate_models(X_test, y_test)
        
        best_models = self.ml_detector.get_best_models(ml_evaluation)
        
        print("  Evaluating ML ensemble...")
        ensemble_pred = self.ml_detector.ensemble_predict(X_test, method='majority_vote')
        ensemble_scores = self.ml_detector.decision_function(X_test)
        
        best_ensemble_score = np.zeros(len(X_test))
        for model_name in best_models.values():
            if model_name in ensemble_scores:
                best_ensemble_score += ensemble_scores[model_name]
        best_ensemble_score /= len(best_models)
        
        print("  Evaluating sequence detector...")
        sequence_scores = self.sequence_detector.predict_anomaly_scores(messages_test)
        sequence_pred = (sequence_scores > 0.7).astype(int)
        
        print("  Evaluating hybrid models...")
        
        hybrid_detector = HybridAnomalyDetector(self.ml_detector, rule_weight=0.4)
        hybrid_detector.add_rule_scores(rule_scores_test)
        hybrid_pred = hybrid_detector.predict_hybrid(X_test, method='weighted_average')
        
        ml_seq_scores = 0.6 * best_ensemble_score + 0.4 * sequence_scores
        ml_seq_pred = (ml_seq_scores < np.percentile(ml_seq_scores, 90)).astype(int)
        
        full_hybrid_scores = (0.4 * best_ensemble_score + 
                              0.3 * (rule_scores_test / 100.0) + 
                              0.3 * sequence_scores)
        full_hybrid_pred = (full_hybrid_scores > 0.5).astype(int)
        
        all_predictions = {
            'ML_Ensemble': ensemble_pred,
            'ML_Best_Ensemble': (best_ensemble_score < np.percentile(best_ensemble_score, 90)).astype(int),
            'Sequence_Detector': sequence_pred,
            'Hybrid_ML_Rules': hybrid_pred,
            'Hybrid_ML_Sequence': ml_seq_pred,
            'Hybrid_Full': full_hybrid_pred
        }
        
        all_scores = {
            'ML_Ensemble': best_ensemble_score,
            'ML_Best_Ensemble': best_ensemble_score,
            'Sequence_Detector': sequence_scores,
            'Hybrid_ML_Rules': full_hybrid_scores,
            'Hybrid_ML_Sequence': ml_seq_scores,
            'Hybrid_Full': full_hybrid_scores
        }
        
        comparison_results = self.evaluator.compare_models(y_test, all_predictions, all_scores)
        
        self.model_results = {
            'ml_evaluation': ml_evaluation,
            'best_models': best_models,
            'all_predictions': all_predictions,
            'all_scores': all_scores,
            'test_labels': y_test,
            'test_split': test_split
        }
        
        self.comparison_results = comparison_results
        
        return self.model_results
    
    def generate_comprehensive_report(self) -> str:
        print("Generating comprehensive report...")
        
        report_generator = ModelComparisonReport(self.evaluator)
        report = report_generator.generate_text_report()
        
        additional_info = []
        additional_info.append("\n" + "="*80)
        additional_info.append("ADVANCED PIPELINE CONFIGURATION")
        additional_info.append("="*80)
        additional_info.append(f"Contamination Rates: {self.contamination_rates}")
        additional_info.append(f"Sequence Length: {self.sequence_length}")
        additional_info.append(f"Batch Size: {self.batch_size}")
        additional_info.append(f"Text Vectorizer: TF-IDF with SVD")
        additional_info.append(f"Feature Types: Temporal, Frequency, Structural, Sequence")
        
        if self.training_data:
            additional_info.append(f"\nTRAINING DATA SUMMARY:")
            additional_info.append(f"Total Messages: {len(self.training_data['messages'])}")
            additional_info.append(f"Feature Dimensions: {self.training_data['combined_features'].shape}")
            additional_info.append(f"Anomaly Rate: {np.mean(self.training_data['labels']):.4f}")
        
        if self.model_results and 'best_models' in self.model_results:
            additional_info.append(f"\nBEST ML MODELS:")
            for algo_type, model_name in self.model_results['best_models'].items():
                additional_info.append(f"  {algo_type}: {model_name}")
        
        return report + "\n".join(additional_info)
    
    def save_all_results(self, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Saving results to {output_dir}...")
        
        models_dir = output_path / "models"
        self.ml_detector.save_models(str(models_dir))
        self.sequence_detector.save_models(str(models_dir))
        
        self.text_vectorizer.save(str(models_dir / "text_vectorizer.pkl"))
        
        self.evaluator.save_results(str(output_path / "evaluation_results.json"))
        
        report = self.generate_comprehensive_report()
        with open(output_path / "comprehensive_report.txt", 'w') as f:
            f.write(report)
        
        training_summary = {
            'total_messages': len(self.training_data.get('messages', [])),
            'feature_shape': self.training_data.get('combined_features', np.array([])).shape,
            'anomaly_rate': float(np.mean(self.training_data.get('labels', []))),
            'config': {
                'contamination_rates': self.contamination_rates,
                'sequence_length': self.sequence_length,
                'batch_size': self.batch_size
            }
        }
        
        with open(output_path / "training_summary.json", 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"All results saved to {output_dir}")


def main():
    print("="*80)
    print("ADVANCED LOG ANOMALY DETECTION PIPELINE")
    print("="*80)
    
    pipeline = AdvancedAnomalyPipeline(
        contamination_rates=[0.02, 0.05, 0.1],
        sequence_length=10,
        batch_size=1000
    )
    
    dataset_dir = project_root / "datasets"
    dataset_files = [
        str(dataset_dir / "comprehensive_logs_50k.txt"),
        str(dataset_dir / "hotel_management_logs.txt"),
        str(dataset_dir / "log_example.txt")
    ]
    
    messages, records = pipeline.load_and_preprocess_data(dataset_files)
    
    if len(messages) < 1000:
        print("Error: Insufficient data for training")
        return
    
    features, parsed_logs = pipeline.extract_features_batched(messages)
    
    text_features = pipeline.vectorize_text(messages)
    
    labels = pipeline.create_labels(parsed_logs)
    
    pipeline.train_models(features, text_features, messages, labels)
    
    results = pipeline.evaluate_all_models(test_split=0.2)
    
    report = pipeline.generate_comprehensive_report()
    print(report)
    
    output_dir = project_root / "backend" / "advanced_results"
    pipeline.save_all_results(str(output_dir))
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    main()
