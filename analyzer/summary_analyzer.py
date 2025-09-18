"""
Summary analysis and evaluation module
요약 결과 분석 및 평가 모듈
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from collections import defaultdict


class SummaryAnalyzer:
    """
    요약 결과 분석 클래스
    eval_summary.py의 기능을 모듈화하여 제공
    """
    
    def __init__(self):
        """Initialize Summary Analyzer"""
        self.labels = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
        self.labels_display = ['MAL', 'SYS', 'IND', 'VUL', 'ORG']
    
    def load_data(
        self, 
        original_path: str, 
        summary_path: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        원본 및 요약 데이터 로드
        
        Args:
            original_path (str): Path to original NER data
            summary_path (str): Path to summary NER data
            
        Returns:
            Tuple[Dict, Dict]: (original_data, summary_data)
        """
        try:
            with open(original_path, "r", encoding="utf-8") as f:
                original_data = json.load(f)
            
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
            
            return original_data, summary_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return {}, {}
    
    def evaluate_model(
        self,
        original_data: Dict[str, Any],
        summary_data: Dict[str, Any],
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        모델 평가 수행
        
        Args:
            original_data (Dict): Original NER data
            summary_data (Dict): Summary NER data
            model_name (str): Name of the model being evaluated
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Initialize statistics containers
        label_freq_stats = {label: {'y_true': 0, 'y_pred': 0} for label in self.labels}
        label_stats = {label: {'y_true': [], 'y_pred': []} for label in self.labels}
        doc_stats = {}
        loss_vectors = []
        false_positives = {}
        
        for doc_id in summary_data:
            if doc_id not in original_data:
                continue
                
            orig = original_data[doc_id]
            summ = summary_data[doc_id]
            
            # Document-level true/pred for metrics
            doc_y_true = []
            doc_y_pred = []
            
            for label in self.labels:
                true_entities = set(orig.get(label, []))
                pred_entities = set(summ.get(label, []))
                
                # Handle special preprocessing for certain models
                if "3.5" in model_name.lower():
                    # Handle comma-separated entities
                    preprocessed_pred_entities = set(summ.get(label, []))
                    pred_entities = set()
                    for entity in preprocessed_pred_entities:
                        pred_entities.update(set(map(lambda x: x.strip(), entity.split(','))))
                
                # False Positives storage
                fp_entities = pred_entities - true_entities
                if fp_entities:
                    if doc_id not in false_positives:
                        false_positives[doc_id] = {}
                    false_positives[doc_id][label] = list(fp_entities)
                
                # Calculate metrics for true positives
                for true_entity in true_entities:
                    label_stats[label]['y_true'].append(1)
                    label_stats[label]['y_pred'].append(1 if true_entity in pred_entities else 0)
                    label_freq_stats[label]['y_true'] += 1
                    label_freq_stats[label]['y_pred'] += 1 if true_entity in pred_entities else 0
                    doc_y_true.append(1)
                    doc_y_pred.append(1 if true_entity in pred_entities else 0)
                
                # Calculate metrics for false positives
                for pred_entity in pred_entities:
                    if pred_entity not in true_entities:
                        label_stats[label]['y_true'].append(0)
                        label_stats[label]['y_pred'].append(1)
                        label_freq_stats[label]['y_pred'] += 1
                        doc_y_true.append(0)
                        doc_y_pred.append(1)
            
            # Calculate document-level loss
            orig_counts = np.array([len(orig.get(label, [])) for label in self.labels])
            summ_counts = np.array([len(summ.get(label, [])) for label in self.labels])
            
            # Normalize counts
            orig_total = orig_counts.sum()
            summ_total = summ_counts.sum()
            
            orig_ratio = orig_counts / orig_total if orig_total != 0 else np.zeros(len(self.labels))
            summ_ratio = summ_counts / summ_total if summ_total != 0 else np.zeros(len(self.labels))
            
            loss = np.abs(orig_ratio - summ_ratio)
            loss_vectors.append(loss)
            
            # Document-level precision/recall/f1
            if doc_y_true:  # Only if there's data
                precision, recall, f1, _ = precision_recall_fscore_support(
                    doc_y_true, doc_y_pred, average='binary', zero_division=0
                )
                doc_stats[doc_id] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                }
        
        return {
            'label_stats': label_stats,
            'label_freq_stats': label_freq_stats,
            'doc_stats': doc_stats,
            'loss_vectors': loss_vectors,
            'false_positives': false_positives,
            'model_name': model_name
        }
    
    def calculate_metrics(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        평가 메트릭 계산
        
        Args:
            evaluation_results (Dict): Results from evaluate_model
            
        Returns:
            Dict[str, Any]: Calculated metrics
        """
        label_stats = evaluation_results['label_stats']
        label_freq_stats = evaluation_results['label_freq_stats']
        
        metrics = {
            'label_metrics': {},
            'overall_metrics': {},
            'label_frequencies': label_freq_stats
        }
        
        # Label-wise metrics
        total_true = []
        total_pred = []
        
        for label in self.labels:
            y_true = label_stats[label]['y_true']
            y_pred = label_stats[label]['y_pred']
            
            if y_true:  # Only calculate if there's data
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )
                
                metrics['label_metrics'][label] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'fdr': fdr,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn
                }
                
                total_true.extend(y_true)
                total_pred.extend(y_pred)
        
        # Overall metrics
        if total_true:
            t_prec, t_recall, t_f1, _ = precision_recall_fscore_support(
                total_true, total_pred, average='binary', zero_division=0
            )
            total_tn, total_fp, total_fn, total_tp = confusion_matrix(total_true, total_pred).ravel()
            total_fdr = total_fp / (total_fp + total_tp) if (total_fp + total_tp) > 0 else 0
            
            metrics['overall_metrics'] = {
                'precision': t_prec,
                'recall': t_recall,
                'f1': t_f1,
                'fdr': total_fdr
            }
        
        return metrics
    
    def print_evaluation_results(
        self, 
        metrics: Dict[str, Any], 
        model_name: str
    ):
        """
        평가 결과 출력
        
        Args:
            metrics (Dict): Calculated metrics
            model_name (str): Model name
        """
        print()
        print("-" * 50)
        print(f"\nEvaluate Dataset: {model_name}")
        
        # Label-wise metrics
        print("Label-wise Metrics:")
        for label in self.labels:
            if label in metrics['label_metrics']:
                m = metrics['label_metrics'][label]
                print(f"🔹 {label}")
                print(f"   Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, "
                      f"F1: {m['f1']:.4f}, FDR: {m['fdr']:.4f}")
        
        # Overall metrics
        if metrics['overall_metrics']:
            om = metrics['overall_metrics']
            print(f"Total Precision: {om['precision']:.4f}, Total Recall: {om['recall']:.4f}, "
                  f"Total F1: {om['f1']:.4f}, Total FDR: {om['fdr']:.4f}")
        
        # Label frequencies
        print("\nLabel Frequency:")
        for label in self.labels:
            if label in metrics['label_frequencies']:
                freq = metrics['label_frequencies'][label]
                print(f"{label} - 원문: {freq['y_true']}개, 요약문: {freq['y_pred']}개")
    
    def create_comparison_dataframe(
        self, 
        multiple_evaluation_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        여러 모델 비교를 위한 데이터프레임 생성
        
        Args:
            multiple_evaluation_results (Dict): Results from multiple models
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_name, results in multiple_evaluation_results.items():
            metrics = self.calculate_metrics(results)
            
            # Overall metrics
            if metrics['overall_metrics']:
                row = {
                    'model': model_name,
                    'overall_precision': metrics['overall_metrics']['precision'],
                    'overall_recall': metrics['overall_metrics']['recall'],
                    'overall_f1': metrics['overall_metrics']['f1'],
                    'overall_fdr': metrics['overall_metrics']['fdr']
                }
                
                # Label-wise metrics
                for label in self.labels:
                    if label in metrics['label_metrics']:
                        lm = metrics['label_metrics'][label]
                        row[f'{label.lower()}_precision'] = lm['precision']
                        row[f'{label.lower()}_recall'] = lm['recall']
                        row[f'{label.lower()}_f1'] = lm['f1']
                        row[f'{label.lower()}_fdr'] = lm['fdr']
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def save_false_positives(
        self, 
        false_positives: Dict[str, Dict[str, List[str]]], 
        output_path: str
    ):
        """
        False Positive 결과 저장
        
        Args:
            false_positives (Dict): False positive data
            output_path (str): Output file path
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(false_positives, f, indent=4, ensure_ascii=False)
            print(f"False positives saved to: {output_path}")
        except Exception as e:
            print(f"Error saving false positives: {e}")
    
    def get_document_statistics(
        self, 
        doc_stats: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        문서별 통계 데이터프레임 생성
        
        Args:
            doc_stats (Dict): Document-level statistics
            
        Returns:
            pd.DataFrame: Document statistics dataframe
        """
        df_doc = pd.DataFrame.from_dict(doc_stats, orient='index')
        df_doc.index.name = 'doc_id'
        df_doc.sort_index(inplace=True)
        return df_doc


if __name__ == "__main__":
    # Test the analyzer
    analyzer = SummaryAnalyzer()
    
    # Example data (would normally load from files)
    example_original = {
        "0": {
            "Malware": ["Zeus"],
            "System": ["Windows"],
            "Indicator": ["network traffic"],
            "Vulnerability": ["banking vulnerability"],
            "Organization": ["cybercriminal group"]
        }
    }
    
    example_summary = {
        "0": {
            "Malware": ["Zeus"],
            "System": ["Windows"],
            "Indicator": ["network traffic"],
            "Vulnerability": [],
            "Organization": ["cybercriminal group"]
        }
    }
    
    # Evaluate
    results = analyzer.evaluate_model(example_original, example_summary, "test_model")
    metrics = analyzer.calculate_metrics(results)
    
    # Print results
    analyzer.print_evaluation_results(metrics, "test_model")
    
    print("Summary analyzer test completed.")
