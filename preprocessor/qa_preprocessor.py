"""
QA data preprocessing for evaluation
QA 데이터의 전처리 및 평가 준비
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import re


class QAPreprocessor:
    """
    QA 데이터 전처리 클래스
    Question Answering 결과의 분석 및 평가 준비
    """
    
    def __init__(self):
        """Initialize QA Preprocessor"""
        self.qa_labels = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
    
    def parse_qa_string(self, qa_string: str) -> List[str]:
        """
        QA 결과 문자열 파싱
        
        Args:
            qa_string (str): Raw QA response string
            
        Returns:
            List[str]: Parsed answers
        """
        try:
            # Remove curly braces and clean the string
            clean_string = qa_string.strip("{} ")
            
            # Split by comma and clean each item
            answers = [item.strip() for item in clean_string.split(',')]
            
            # Ensure we have exactly 5 answers
            while len(answers) < len(self.qa_labels):
                answers.append("no answer")
            
            return answers[:len(self.qa_labels)]
            
        except Exception as e:
            print(f"Error parsing QA string: {e}")
            return ["no answer"] * len(self.qa_labels)
    
    def preprocess_qa_results(
        self, 
        qa_results: List[Dict[str, Any]]
    ) -> Dict[int, List[str]]:
        """
        QA 결과 전처리
        
        Args:
            qa_results (List[Dict[str, Any]]): Raw QA results
            
        Returns:
            Dict[int, List[str]]: Preprocessed QA data {id: [answers]}
        """
        processed_qa = {}
        
        for result in qa_results:
            doc_id = result.get("id", 0)
            qa_text = result.get("qa_text", "")
            
            # Parse QA response
            parsed_answers = self.parse_qa_string(qa_text)
            processed_qa[doc_id] = parsed_answers
        
        return processed_qa
    
    def evaluate_qa_vs_ner(
        self,
        processed_qa: Dict[int, List[str]],
        original_ner_dict: Dict[str, Dict[str, List[str]]],
        swap_indicator_vulnerability: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        QA 결과와 원본 NER 데이터 비교 평가
        
        Args:
            processed_qa (Dict): Processed QA results
            original_ner_dict (Dict): Original NER data
            swap_indicator_vulnerability (bool): Whether to swap indicator and vulnerability
            
        Returns:
            Dict[str, Dict[str, Any]]: Evaluation results
        """
        evaluation_results = {
            "label_metrics": {label: {"y_true": [], "y_pred": []} for label in self.qa_labels},
            "failed_predictions": [],
            "statistics": {"total_processed": 0, "total_failed": 0}
        }
        
        for doc_id, pred_entities in processed_qa.items():
            ner_entry = original_ner_dict.get(str(doc_id))
            if not ner_entry:
                continue
            
            evaluation_results["statistics"]["total_processed"] += 1
            
            try:
                for idx, label in enumerate(self.qa_labels):
                    # Handle indicator/vulnerability swap if requested
                    if swap_indicator_vulnerability:
                        if idx == 2:  # Indicator -> Vulnerability
                            pred = pred_entities[3] if len(pred_entities) > 3 else "no answer"
                        elif idx == 3:  # Vulnerability -> Indicator
                            pred = pred_entities[2] if len(pred_entities) > 2 else "no answer"
                        else:
                            pred = pred_entities[idx] if len(pred_entities) > idx else "no answer"
                    else:
                        pred = pred_entities[idx] if len(pred_entities) > idx else "no answer"
                    
                    # Get gold entities (case-insensitive comparison)
                    gold_entities = set(map(str.lower, ner_entry.get(label, [])))
                    
                    # Process prediction
                    pred_set = set() if pred.lower() == "no answer" else {pred.lower()}
                    
                    # Check if prediction matches any gold entity
                    match = bool(pred_set & gold_entities)
                    
                    # Record results for metrics calculation
                    evaluation_results["label_metrics"][label]["y_true"].append(1 if match else 0)
                    evaluation_results["label_metrics"][label]["y_pred"].append(1 if pred_set else 0)
                    
                    # Record failed predictions for analysis
                    if not match and gold_entities:
                        evaluation_results["failed_predictions"].append({
                            "doc_id": doc_id,
                            "label": label,
                            "prediction": pred,
                            "gold_entities": list(gold_entities),
                            "match": match
                        })
                        
            except Exception as e:
                print(f"Error evaluating document {doc_id}: {e}")
                evaluation_results["statistics"]["total_failed"] += 1
                continue
        
        return evaluation_results
    
    def calculate_qa_metrics(
        self, 
        evaluation_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        QA 평가 메트릭 계산
        
        Args:
            evaluation_results (Dict): Results from evaluate_qa_vs_ner
            
        Returns:
            Dict[str, Dict[str, float]]: Calculated metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {}
        
        for label in self.qa_labels:
            y_true = evaluation_results["label_metrics"][label]["y_true"]
            y_pred = evaluation_results["label_metrics"][label]["y_pred"]
            
            if not y_true:  # Skip if no data
                continue
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "total_samples": len(y_true)
            }
        
        # Calculate overall metrics
        all_y_true = []
        all_y_pred = []
        
        for label in self.qa_labels:
            all_y_true.extend(evaluation_results["label_metrics"][label]["y_true"])
            all_y_pred.extend(evaluation_results["label_metrics"][label]["y_pred"])
        
        if all_y_true:
            metrics["overall"] = {
                "precision": precision_score(all_y_true, all_y_pred, zero_division=0),
                "recall": recall_score(all_y_true, all_y_pred, zero_division=0),
                "f1": f1_score(all_y_true, all_y_pred, zero_division=0),
                "total_samples": len(all_y_true)
            }
        
        return metrics
    
    def save_failed_predictions(
        self, 
        failed_predictions: List[Dict[str, Any]], 
        output_path: str
    ):
        """
        실패한 예측 결과 저장
        
        Args:
            failed_predictions (List[Dict]): Failed prediction data
            output_path (str): Output CSV file path
        """
        import csv
        
        try:
            with open(output_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["doc_id", "label", "prediction", "gold_entities", "match"])
                
                for failure in failed_predictions:
                    writer.writerow([
                        failure["doc_id"],
                        failure["label"],
                        failure["prediction"],
                        '; '.join(failure["gold_entities"]),
                        failure["match"]
                    ])
            
            print(f"Failed predictions saved to: {output_path}")
            
        except Exception as e:
            print(f"Error saving failed predictions: {e}")
    
    def load_original_ner_data(self, file_path: str) -> Dict[str, Dict[str, List[str]]]:
        """
        원본 NER 데이터 로드
        
        Args:
            file_path (str): Path to original NER data file
            
        Returns:
            Dict[str, Dict[str, List[str]]]: Original NER data
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading original NER data: {e}")
            return {}
    
    def create_evaluation_report(
        self,
        metrics: Dict[str, Dict[str, float]],
        evaluation_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        평가 리포트 생성
        
        Args:
            metrics (Dict): Calculated metrics
            evaluation_results (Dict): Evaluation results
            
        Returns:
            str: Formatted evaluation report
        """
        report = []
        report.append("=== QA Evaluation Report ===")
        report.append("")
        
        # Overall statistics
        stats = evaluation_results["statistics"]
        report.append(f"Total documents processed: {stats['total_processed']}")
        report.append(f"Total failed: {stats['total_failed']}")
        report.append(f"Success rate: {(stats['total_processed'] - stats['total_failed']) / stats['total_processed'] * 100:.2f}%")
        report.append("")
        
        # Label-wise metrics
        report.append("Label-wise Metrics:")
        for label in self.qa_labels:
            if label in metrics:
                m = metrics[label]
                report.append(f"🔹 {label}")
                report.append(f"   Precision: {m['precision']:.4f}")
                report.append(f"   Recall: {m['recall']:.4f}")
                report.append(f"   F1: {m['f1']:.4f}")
                report.append(f"   Total samples: {m['total_samples']}")
                report.append("")
        
        # Overall metrics
        if "overall" in metrics:
            m = metrics["overall"]
            report.append("Overall Metrics:")
            report.append(f"   Precision: {m['precision']:.4f}")
            report.append(f"   Recall: {m['recall']:.4f}")
            report.append(f"   F1: {m['f1']:.4f}")
            report.append(f"   Total samples: {m['total_samples']}")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test the QA preprocessor
    preprocessor = QAPreprocessor()
    
    # Example QA results
    example_qa_results = [
        {
            "id": 0,
            "qa_text": "{Zeus, Windows, banking vulnerability, network traffic, cybercriminal group}"
        },
        {
            "id": 1,
            "qa_text": "{Stuxnet, Windows, SCADA vulnerability, no answer, nation-state}"
        }
    ]
    
    # Preprocess QA results
    processed_qa = preprocessor.preprocess_qa_results(example_qa_results)
    print(f"Processed QA: {processed_qa}")
    
    # Example original NER data
    example_ner = {
        "0": {
            "Malware": ["Zeus"],
            "System": ["Windows"],
            "Vulnerability": ["banking vulnerability"],
            "Indicator": ["network traffic"],
            "Organization": ["cybercriminal group"]
        }
    }
    
    # Evaluate
    evaluation_results = preprocessor.evaluate_qa_vs_ner(processed_qa, example_ner)
    metrics = preprocessor.calculate_qa_metrics(evaluation_results)
    
    print("QA evaluation metrics:", metrics)
    print("QA preprocessor test completed.")
