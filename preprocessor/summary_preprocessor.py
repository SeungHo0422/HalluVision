"""
Summary data preprocessing for evaluation
요약 데이터의 전처리 및 평가 준비
"""

import json
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import re


class SummaryPreprocessor:
    """
    요약 데이터 전처리 클래스
    NER 평가를 위한 데이터 준비 및 형식 변환
    """
    
    def __init__(self):
        """Initialize Summary Preprocessor"""
        self.ner_labels = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
    
    def extract_entities_from_summary(
        self, 
        summary_text: str, 
        model_type: str = "cyner"
    ) -> Dict[str, List[str]]:
        """
        요약문에서 엔티티 추출 (실제 NER 모델 연동 필요)
        
        Args:
            summary_text (str): Summary text
            model_type (str): NER model type ("cyner", "spacy", etc.)
            
        Returns:
            Dict[str, List[str]]: Extracted entities by label
        """
        # TODO: 실제 NER 모델 연동 필요
        # 현재는 더미 구현
        
        entities = {label: [] for label in self.ner_labels}
        
        # Simple keyword-based extraction (placeholder)
        malware_keywords = ['trojan', 'virus', 'malware', 'ransomware', 'spyware', 'rootkit']
        system_keywords = ['windows', 'linux', 'android', 'ios', 'macos']
        
        summary_lower = summary_text.lower()
        
        for keyword in malware_keywords:
            if keyword in summary_lower:
                entities["Malware"].append(keyword.capitalize())
        
        for keyword in system_keywords:
            if keyword in summary_lower:
                entities["System"].append(keyword.capitalize())
        
        return entities
    
    def prepare_for_evaluation(
        self, 
        summary_results: List[Dict[str, Any]],
        original_ner_data: Optional[Dict[str, Dict[str, List[str]]]] = None
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        평가를 위한 요약 데이터 준비
        
        Args:
            summary_results (List[Dict[str, Any]]): Summary results from LLM
            original_ner_data (Optional[Dict]): Original NER data for comparison
            
        Returns:
            Dict[str, Dict[str, List[str]]]: Prepared data for evaluation
        """
        prepared_data = {}
        
        for result in summary_results:
            doc_id = str(result.get("id", 0))
            summary_text = result.get("summary_text", "")
            
            # Extract entities from summary
            extracted_entities = self.extract_entities_from_summary(summary_text)
            
            prepared_data[doc_id] = extracted_entities
        
        return prepared_data
    
    def load_original_ner_data(self, file_path: str) -> Dict[str, Dict[str, List[str]]]:
        """
        원본 NER 데이터 로드
        
        Args:
            file_path (str): Path to original NER data
            
        Returns:
            Dict[str, Dict[str, List[str]]]: Original NER data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading original NER data: {e}")
            return {}
    
    def calculate_entity_statistics(
        self, 
        summary_ner_data: Dict[str, Dict[str, List[str]]],
        original_ner_data: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, Any]:
        """
        엔티티 통계 계산
        
        Args:
            summary_ner_data (Dict): NER data from summaries
            original_ner_data (Dict): Original NER data
            
        Returns:
            Dict[str, Any]: Entity statistics
        """
        stats = {
            "label_frequencies": {label: {"original": 0, "summary": 0} for label in self.ner_labels},
            "document_coverage": {},
            "entity_overlap": {}
        }
        
        # Calculate label frequencies
        for doc_id in original_ner_data:
            orig_data = original_ner_data[doc_id]
            summ_data = summary_ner_data.get(doc_id, {})
            
            for label in self.ner_labels:
                orig_entities = orig_data.get(label, [])
                summ_entities = summ_data.get(label, [])
                
                stats["label_frequencies"][label]["original"] += len(orig_entities)
                stats["label_frequencies"][label]["summary"] += len(summ_entities)
                
                # Calculate overlap
                if orig_entities or summ_entities:
                    overlap = len(set(orig_entities) & set(summ_entities))
                    total_unique = len(set(orig_entities) | set(summ_entities))
                    
                    if doc_id not in stats["entity_overlap"]:
                        stats["entity_overlap"][doc_id] = {}
                    
                    stats["entity_overlap"][doc_id][label] = {
                        "overlap": overlap,
                        "total_unique": total_unique,
                        "overlap_ratio": overlap / total_unique if total_unique > 0 else 0
                    }
        
        return stats
    
    def preprocess_for_model_comparison(
        self, 
        model_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """
        모델 간 비교를 위한 전처리
        
        Args:
            model_results (Dict): Results from different models
            
        Returns:
            Dict: Preprocessed data for model comparison
        """
        comparison_data = {}
        
        for model_name, results in model_results.items():
            model_data = {}
            
            for result in results:
                doc_id = str(result.get("id", 0))
                summary_text = result.get("summary_text", "")
                
                # Extract entities
                entities = self.extract_entities_from_summary(summary_text)
                model_data[doc_id] = entities
            
            comparison_data[model_name] = model_data
        
        return comparison_data
    
    def save_preprocessed_data(
        self, 
        data: Dict[str, Any], 
        output_path: str
    ):
        """
        전처리된 데이터 저장
        
        Args:
            data (Dict[str, Any]): Preprocessed data
            output_path (str): Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Preprocessed data saved to: {output_path}")
        except Exception as e:
            print(f"Error saving preprocessed data: {e}")
    
    def create_evaluation_matrices(
        self,
        summary_ner_data: Dict[str, Dict[str, List[str]]],
        original_ner_data: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        평가 매트릭스 생성 (precision, recall 계산용)
        
        Args:
            summary_ner_data (Dict): NER data from summaries
            original_ner_data (Dict): Original NER data
            
        Returns:
            Dict[str, Dict[str, List[int]]]: Evaluation matrices
        """
        matrices = {label: {"y_true": [], "y_pred": []} for label in self.ner_labels}
        
        for doc_id in original_ner_data:
            orig_data = original_ner_data[doc_id]
            summ_data = summary_ner_data.get(doc_id, {})
            
            for label in self.ner_labels:
                orig_entities = set(orig_data.get(label, []))
                summ_entities = set(summ_data.get(label, []))
                
                # True positives and false positives
                for entity in orig_entities:
                    matrices[label]["y_true"].append(1)
                    matrices[label]["y_pred"].append(1 if entity in summ_entities else 0)
                
                # False positives
                for entity in summ_entities - orig_entities:
                    matrices[label]["y_true"].append(0)
                    matrices[label]["y_pred"].append(1)
        
        return matrices


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = SummaryPreprocessor()
    
    # Example summary results
    example_results = [
        {
            "id": 0,
            "summary_text": "Zeus is a banking trojan that targets Windows systems and steals credentials.",
            "model": "gpt-3.5-turbo"
        },
        {
            "id": 1,
            "summary_text": "Stuxnet is a sophisticated malware that attacked industrial control systems.",
            "model": "gpt-3.5-turbo"
        }
    ]
    
    # Prepare for evaluation
    prepared_data = preprocessor.prepare_for_evaluation(example_results)
    print(f"Prepared data: {prepared_data}")
    
    # Calculate statistics (would need original data in practice)
    print("Summary preprocessor test completed.")
