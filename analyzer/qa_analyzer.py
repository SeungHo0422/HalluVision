"""
QA analysis and evaluation module
QA 결과 분석 및 평가 모듈 (analysis_qa.py 기반)
"""

import json
import csv
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict


class QAAnalyzer:
    """
    QA 결과 분석 클래스
    analysis_qa.py의 기능을 모듈화하여 제공
    """
    
    def __init__(self):
        """Initialize QA Analyzer"""
        self.labels = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
    
    def load_data(
        self, 
        original_path: str, 
        qa_data_path: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        원본 NER 데이터와 QA 결과 데이터 로드
        
        Args:
            original_path (str): Path to original NER data
            qa_data_path (str): Path to QA results data
            
        Returns:
            Tuple[Dict, List]: (original_data, qa_data)
        """
        try:
            with open(original_path, "r", encoding="utf-8") as f:
                original_data = json.load(f)
            
            with open(qa_data_path, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
            
            return original_data, qa_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return {}, []
    
    def preprocess_qa_data(self, qa_data: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        QA 데이터 전처리
        
        Args:
            qa_data (List[Dict]): Raw QA data
            
        Returns:
            Dict[int, List[str]]: Processed QA data {id: [answers]}
        """
        processed_qa = {}
        
        for item in qa_data:
            if "id" in item and "qa_text" in item:
                doc_id = item["id"]
                qa_text = item["qa_text"]
                
                # Parse QA response
                items = qa_text.strip("{} ").split(',')
                items = [item.strip() for item in items if item.strip()]
                
                # Ensure we have 5 answers
                while len(items) < len(self.labels):
                    items.append("no answer")
                
                processed_qa[doc_id] = items[:len(self.labels)]
        
        return processed_qa
    
    def evaluate_qa_vs_ner(
        self,
        processed_qa: Dict[int, List[str]],
        original_ner_dict: Dict[str, Dict[str, List[str]]],
        swap_indicator_vulnerability: bool = True
    ) -> Dict[str, Any]:
        """
        QA 결과와 NER 데이터 비교 평가
        
        Args:
            processed_qa (Dict): Processed QA data
            original_ner_dict (Dict): Original NER data
            swap_indicator_vulnerability (bool): Whether to swap indicator and vulnerability indices
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        y_true = defaultdict(list)
        y_pred = defaultdict(list)
        failed_predictions = []
        
        for id_, pred_entities in processed_qa.items():
            ner_entry = original_ner_dict.get(str(id_))
            if not ner_entry:
                continue
            
            try:
                for idx, label in enumerate(self.labels):
                    # Handle indicator/vulnerability swap
                    if swap_indicator_vulnerability:
                        if idx == 2:  # Indicator -> use vulnerability position
                            pred = pred_entities[3] if len(pred_entities) > 3 else "no answer"
                        elif idx == 3:  # Vulnerability -> use indicator position
                            pred = pred_entities[2] if len(pred_entities) > 2 else "no answer"
                        else:
                            pred = pred_entities[idx] if len(pred_entities) > idx else "no answer"
                    else:
                        pred = pred_entities[idx] if len(pred_entities) > idx else "no answer"
                    
                    # Get gold entities (case-insensitive)
                    gold_entities = set(map(str.lower, ner_entry.get(label, [])))
                    
                    # Process prediction
                    pred_set = set() if pred.lower() == "no answer" else {pred.lower()}
                    
                    # Check if prediction matches gold
                    match = bool(pred_set & gold_entities)
                    
                    # Store for metrics calculation
                    y_true[label].append(1 if match else 0)
                    y_pred[label].append(1 if pred_set else 0)
                    
                    # Record failed predictions
                    if not match and gold_entities:
                        failed_predictions.append({
                            "id": id_,
                            "label": label,
                            "prediction": pred,
                            "gold_entities": list(gold_entities),
                            "match": match
                        })
                        
            except Exception as e:
                print(f"Error processing document {id_}: {e}")
                continue
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'failed_predictions': failed_predictions
        }
    
    def calculate_metrics(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        평가 메트릭 계산
        
        Args:
            evaluation_results (Dict): Results from evaluate_qa_vs_ner
            
        Returns:
            Dict[str, Any]: Calculated metrics
        """
        y_true = evaluation_results['y_true']
        y_pred = evaluation_results['y_pred']
        
        metrics = {
            'label_metrics': {},
            'overall_metrics': {}
        }
        
        # Label-wise metrics
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        for label in self.labels:
            if label in y_true and y_true[label]:
                precision = precision_score(y_true[label], y_pred[label], zero_division=0)
                recall = recall_score(y_true[label], y_pred[label], zero_division=0)
                f1 = f1_score(y_true[label], y_pred[label], zero_division=0)
                
                metrics['label_metrics'][label] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'total_samples': len(y_true[label])
                }
                
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1s.append(f1)
        
        # Overall averages
        if all_precisions:
            metrics['overall_metrics'] = {
                'avg_precision': sum(all_precisions) / len(all_precisions),
                'avg_recall': sum(all_recalls) / len(all_recalls),
                'avg_f1': sum(all_f1s) / len(all_f1s)
            }
        
        return metrics
    
    def print_evaluation_results(
        self, 
        metrics: Dict[str, Any], 
        total_original: int, 
        total_qa: int
    ):
        """
        평가 결과 출력
        
        Args:
            metrics (Dict): Calculated metrics
            total_original (int): Total original articles
            total_qa (int): Total QA articles
        """
        print("=== QA Evaluation Results ===")
        print(f"Original Articles: {total_original}")
        print(f"LLM QA Articles: {total_qa}")
        
        # Label-wise results
        for label in self.labels:
            if label in metrics['label_metrics']:
                m = metrics['label_metrics'][label]
                print(f"{label:15s} | Precision: {m['precision']:.4f} | "
                      f"Recall: {m['recall']:.4f} | F1: {m['f1']:.4f} | "
                      f"총 개수: {m['total_samples']}")
        
        # Overall averages
        if metrics['overall_metrics']:
            om = metrics['overall_metrics']
            print(f"\n{'문서 총 평균':15s} | Precision: {om['avg_precision']:.4f} | "
                  f"Recall: {om['avg_recall']:.4f} | F1: {om['avg_f1']:.4f}")
    
    def save_failed_predictions(
        self, 
        failed_predictions: List[Dict[str, Any]], 
        output_path: str
    ):
        """
        실패한 예측 결과 저장
        
        Args:
            failed_predictions (List): Failed prediction data
            output_path (str): Output CSV file path
        """
        try:
            with open(output_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "label", "prediction", "gold_entities", "match"])
                
                for failure in failed_predictions:
                    writer.writerow([
                        failure["id"],
                        failure["label"],
                        failure["prediction"],
                        '; '.join(failure["gold_entities"]),
                        failure["match"]
                    ])
            
            print(f"Failed predictions saved to: {output_path}")
            
        except Exception as e:
            print(f"Error saving failed predictions: {e}")
    
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
            
            row = {'model': model_name}
            
            # Overall metrics
            if metrics['overall_metrics']:
                row.update({
                    'avg_precision': metrics['overall_metrics']['avg_precision'],
                    'avg_recall': metrics['overall_metrics']['avg_recall'],
                    'avg_f1': metrics['overall_metrics']['avg_f1']
                })
            
            # Label-wise metrics
            for label in self.labels:
                if label in metrics['label_metrics']:
                    lm = metrics['label_metrics'][label]
                    row[f'{label.lower()}_precision'] = lm['precision']
                    row[f'{label.lower()}_recall'] = lm['recall']
                    row[f'{label.lower()}_f1'] = lm['f1']
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def analyze_failure_patterns(
        self, 
        failed_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        실패 패턴 분석
        
        Args:
            failed_predictions (List): Failed prediction data
            
        Returns:
            Dict[str, Any]: Failure analysis results
        """
        failure_analysis = {
            'label_failure_counts': defaultdict(int),
            'common_failures': defaultdict(list),
            'total_failures': len(failed_predictions)
        }
        
        for failure in failed_predictions:
            label = failure['label']
            prediction = failure['prediction']
            
            failure_analysis['label_failure_counts'][label] += 1
            failure_analysis['common_failures'][label].append(prediction)
        
        # Find most common failed predictions per label
        for label in failure_analysis['common_failures']:
            predictions = failure_analysis['common_failures'][label]
            pred_counts = defaultdict(int)
            
            for pred in predictions:
                pred_counts[pred] += 1
            
            # Sort by frequency
            sorted_failures = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)
            failure_analysis['common_failures'][label] = sorted_failures[:5]  # Top 5
        
        return failure_analysis
    
    def run_full_evaluation(
        self,
        original_data: Dict[str, Any],
        qa_data: List[Dict[str, Any]],
        model_name: str = "model",
        save_failures: bool = True,
        output_dir: str = "output"
    ) -> Dict[str, Any]:
        """
        전체 평가 파이프라인 실행
        
        Args:
            original_data (Dict): Original NER data
            qa_data (List): QA results data
            model_name (str): Model name
            save_failures (bool): Whether to save failure analysis
            output_dir (str): Output directory
            
        Returns:
            Dict[str, Any]: Complete evaluation results
        """
        # Preprocess QA data
        processed_qa = self.preprocess_qa_data(qa_data)
        
        # Evaluate
        evaluation_results = self.evaluate_qa_vs_ner(processed_qa, original_data)
        
        # Calculate metrics
        metrics = self.calculate_metrics(evaluation_results)
        
        # Print results
        self.print_evaluation_results(metrics, len(original_data), len(qa_data))
        
        # Save failed predictions if requested
        if save_failures and evaluation_results['failed_predictions']:
            failure_path = f"{output_dir}/failed_cases_{model_name.lower().replace('-', '_')}.csv"
            self.save_failed_predictions(evaluation_results['failed_predictions'], failure_path)
        
        # Analyze failure patterns
        failure_analysis = self.analyze_failure_patterns(evaluation_results['failed_predictions'])
        
        return {
            'metrics': metrics,
            'evaluation_results': evaluation_results,
            'failure_analysis': failure_analysis,
            'model_name': model_name
        }


if __name__ == "__main__":
    # Test the QA analyzer
    analyzer = QAAnalyzer()
    
    # Example data
    example_original = {
        "0": {
            "Malware": ["Zeus"],
            "System": ["Windows"],
            "Indicator": ["network traffic"],
            "Vulnerability": ["banking vulnerability"],
            "Organization": ["cybercriminal group"]
        }
    }
    
    example_qa_data = [
        {
            "id": 0,
            "qa_text": "{Zeus, Windows, banking vulnerability, network traffic, cybercriminal group}"
        }
    ]
    
    # Run evaluation
    results = analyzer.run_full_evaluation(
        example_original, 
        example_qa_data, 
        "test_model",
        save_failures=False
    )
    
    print("QA analyzer test completed.")
    print(f"Metrics: {results['metrics']}")
