
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedEvaluator:
    
    def __init__(self):
        self.results = {}
        self.comparison_results = {}
        
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'support_positive': int(np.sum(y_true)),
            'support_negative': int(len(y_true) - np.sum(y_true))
        }
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        return {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_samples': len(y_true)
        }
    
    def calculate_advanced_metrics(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = 0.5  
        
        
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            pr_auc = np.trapz(precision, recall)
        except ValueError:
            pr_auc = 0.0
        
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'roc_auc': auc,
            'pr_auc': pr_auc,
            'optimal_threshold': float(optimal_threshold),
            'max_tpr_minus_fpr': float(tpr[optimal_idx] - fpr[optimal_idx])
        }
    
    def evaluate_model(self, 
                      model_name: str,
                      y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      y_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        
        
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
        
        
        conf_matrix = self.calculate_confusion_matrix(y_true, y_pred)
        
        
        advanced_metrics = {}
        if y_scores is not None:
            advanced_metrics = self.calculate_advanced_metrics(y_true, y_scores)
        
        
        evaluation = {
            'model_name': model_name,
            'basic_metrics': basic_metrics,
            'confusion_matrix': conf_matrix,
            'advanced_metrics': advanced_metrics,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        self.results[model_name] = evaluation
        return evaluation
    
    def compare_models(self, y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray], 
                      scores_dict: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        
        comparison = {
            'model_rankings': {},
            'metric_comparison': {},
            'best_models': {},
            'summary_statistics': {}
        }
        
        
        for model_name, y_pred in predictions_dict.items():
            y_scores = scores_dict.get(model_name) if scores_dict else None
            self.evaluate_model(model_name, y_true, y_pred, y_scores)
        
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics:
            model_scores = {}
            for model_name, result in self.results.items():
                if metric in result['basic_metrics']:
                    model_scores[model_name] = result['basic_metrics'][metric]
                elif metric in result['advanced_metrics']:
                    model_scores[model_name] = result['advanced_metrics'][metric]
            
            if model_scores:
                
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
                comparison['model_rankings'][metric] = sorted_models
                comparison['metric_comparison'][metric] = model_scores
                comparison['best_models'][metric] = sorted_models[0][0] if sorted_models else None
        
        
        all_metrics_data = []
        for result in self.results.values():
            model_data = {'model': result['model_name']}
            model_data.update(result['basic_metrics'])
            model_data.update(result['advanced_metrics'])
            all_metrics_data.append(model_data)
        
        comparison['summary_statistics'] = {
            'total_models': len(self.results),
            'metrics_available': list(metrics),
            'model_details': all_metrics_data
        }
        
        self.comparison_results = comparison
        return comparison
    
    def create_comparison_table(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        
        table_data = []
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            row.update(result['basic_metrics'])
            row.update(result['advanced_metrics'])
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        
        column_order = ['Model', 'accuracy', 'precision', 'recall', 'f1_score', 
                       'roc_auc', 'pr_auc', 'support_positive', 'support_negative']
        
        
        available_columns = [col for col in column_order if col in df.columns]
        other_columns = [col for col in df.columns if col not in available_columns]
        final_columns = available_columns + other_columns
        
        return df[final_columns].round(4)
    
    def print_detailed_results(self):
        if not self.results:
            print("No evaluation results available.")
            return
        
        print("\n" + "="*80)
        print("DETAILED MODEL EVALUATION RESULTS")
        print("="*80)
        
        for model_name, result in self.results.items():
            print(f"\n{'='*60}")
            print(f"MODEL: {model_name.upper()}")
            print(f"{'='*60}")
            
            
            print("\nBASIC METRICS:")
            basic = result['basic_metrics']
            print(f"  Accuracy:    {basic['accuracy']:.4f}")
            print(f"  Precision:   {basic['precision']:.4f}")
            print(f"  Recall:      {basic['recall']:.4f}")
            print(f"  F1-Score:    {basic['f1_score']:.4f}")
            print(f"  Support:     {basic['support_positive']} positive, {basic['support_negative']} negative")
            
            
            print("\nCONFUSION MATRIX:")
            conf = result['confusion_matrix']
            print(f"  True Positives:    {conf['true_positives']}")
            print(f"  False Positives:   {conf['false_positives']}")
            print(f"  True Negatives:    {conf['true_negatives']}")
            print(f"  False Negatives:   {conf['false_negatives']}")
            print(f"  Total Samples:     {conf['total_samples']}")
            
            
            if result['advanced_metrics']:
                print("\nADVANCED METRICS:")
                adv = result['advanced_metrics']
                print(f"  ROC-AUC:           {adv['roc_auc']:.4f}")
                print(f"  PR-AUC:            {adv['pr_auc']:.4f}")
                print(f"  Optimal Threshold: {adv['optimal_threshold']:.4f}")
                print(f"  Max(TPR-FPR):      {adv['max_tpr_minus_fpr']:.4f}")
        
        
        if self.comparison_results:
            print(f"\n{'='*80}")
            print("MODEL COMPARISON SUMMARY")
            print(f"{'='*80}")
            
            rankings = self.comparison_results['best_models']
            print("\nBEST MODELS BY METRIC:")
            for metric, best_model in rankings.items():
                if best_model:
                    score = self.comparison_results['metric_comparison'][metric][best_model]
                    print(f"  {metric.upper()}: {best_model} ({score:.4f})")
    
    def save_results(self, filepath: str):
        results_data = {
            'individual_results': self.results,
            'comparison_results': self.comparison_results,
            'export_timestamp': datetime.now().isoformat(),
            'total_models_evaluated': len(self.results)
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"Evaluation results saved to {filepath}")
    
    def load_results(self, filepath: str):
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        self.results = results_data['individual_results']
        self.comparison_results = results_data['comparison_results']
        
        print(f"Evaluation results loaded from {filepath}")
    
    def create_improvement_summary(self, baseline_results: Dict, improved_results: Dict) -> Dict[str, Any]:
        
        summary = {
            'improvements': {},
            'degradations': {},
            'significant_changes': {},
            'overall_assessment': ''
        }
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics:
            baseline_value = 0
            improved_value = 0
            
            
            for result in baseline_results.values():
                if metric in result.get('basic_metrics', {}):
                    baseline_value = result['basic_metrics'][metric]
                elif metric in result.get('advanced_metrics', {}):
                    baseline_value = result['advanced_metrics'][metric]
            
            
            for result in improved_results.values():
                if metric in result.get('basic_metrics', {}):
                    improved_value = result['basic_metrics'][metric]
                elif metric in result.get('advanced_metrics', {}):
                    improved_value = result['advanced_metrics'][metric]
            
            if baseline_value > 0:
                improvement = (improved_value - baseline_value) / baseline_value * 100
                
                if improvement > 0:
                    summary['improvements'][metric] = f"+{improvement:.2f}%"
                    if improvement > 10:
                        summary['significant_changes'][metric] = f"Significant improvement: +{improvement:.2f}%"
                elif improvement < 0:
                    summary['degradations'][metric] = f"{improvement:.2f}%"
                    if improvement < -10:
                        summary['significant_changes'][metric] = f"Significant degradation: {improvement:.2f}%"
        
        
        if len(summary['improvements']) > len(summary['degradations']):
            summary['overall_assessment'] = "Overall improvement achieved"
        elif len(summary['improvements']) < len(summary['degradations']):
            summary['overall_assessment'] = "Overall performance degraded"
        else:
            summary['overall_assessment'] = "Mixed results - trade-offs between metrics"
        
        return summary


class ModelComparisonReport:
    
    def __init__(self, evaluator: AdvancedEvaluator):
        self.evaluator = evaluator
    
    def generate_text_report(self) -> str:
        if not self.evaluator.results:
            return "No evaluation results available."
        
        report = []
        report.append("LOG ANOMALY DETECTION MODEL COMPARISON REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Models Evaluated: {len(self.evaluator.results)}")
        report.append("")
        
        
        df = self.evaluator.create_comparison_table()
        if not df.empty:
            report.append("MODEL COMPARISON TABLE:")
            report.append("-" * 40)
            report.append(df.to_string(index=False))
            report.append("")
        
        
        if self.evaluator.comparison_results:
            best_models = self.evaluator.comparison_results['best_models']
            report.append("BEST MODELS BY METRIC:")
            report.append("-" * 30)
            for metric, model in best_models.items():
                if model and metric in self.evaluator.comparison_results['metric_comparison']:
                    score = self.evaluator.comparison_results['metric_comparison'][metric][model]
                    report.append(f"{metric.upper()}: {model} ({score:.4f})")
            report.append("")
        
        
        report.append("DETAILED MODEL ANALYSIS:")
        report.append("-" * 35)
        
        for model_name, result in self.evaluator.results.items():
            report.append(f"\n{model_name.upper()}:")
            
            basic = result['basic_metrics']
            report.append(f"  Accuracy: {basic['accuracy']:.4f}")
            report.append(f"  Precision: {basic['precision']:.4f}")
            report.append(f"  Recall: {basic['recall']:.4f}")
            report.append(f"  F1-Score: {basic['f1_score']:.4f}")
            
            conf = result['confusion_matrix']
            report.append(f"  TP: {conf['true_positives']}, FP: {conf['false_positives']}")
            report.append(f"  TN: {conf['true_negatives']}, FN: {conf['false_negatives']}")
            
            if result['advanced_metrics']:
                adv = result['advanced_metrics']
                report.append(f"  ROC-AUC: {adv['roc_auc']:.4f}")
                report.append(f"  PR-AUC: {adv['pr_auc']:.4f}")
        
        return "\n".join(report)
    
    def save_report(self, filepath: str):
        report = self.generate_text_report()
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"Comparison report saved to {filepath}")
