"""
Adapter module to connect data scraper output with LLM proxy input
데이터 스크래퍼의 출력과 LLM 프록시의 입력 형식을 연결하는 어댑터
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class DataAdapter:
    """
    데이터 형식 변환을 담당하는 어댑터 클래스
    스크래핑된 데이터를 LLM 처리에 적합한 형식으로 변환
    """
    
    def __init__(self):
        """Initialize DataAdapter"""
        pass
    
    def scraped_to_text_list(self, scraped_data: List[Dict[str, Any]]) -> List[str]:
        """
        스크래핑된 데이터를 텍스트 리스트로 변환
        
        Args:
            scraped_data (List[Dict[str, Any]]): Data from data scraper
            
        Returns:
            List[str]: List of combined text content
        """
        text_list = []
        
        for data in scraped_data:
            content = data.get("combined_content", "")
            if content.strip():
                text_list.append(content)
        
        return text_list
    
    def text_file_to_list(self, file_path: str, delimiter: str = "Share this article:") -> List[str]:
        """
        기존 텍스트 파일을 리스트로 변환 (gpt_response.py 호환)
        
        Args:
            file_path (str): Path to text file
            delimiter (str): Delimiter to split articles
            
        Returns:
            List[str]: List of articles
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                articles = content.split(delimiter)
                return [article.strip() for article in articles if article.strip()]
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
    
    def llm_results_to_standard_format(
        self, 
        llm_results: List[Dict[str, Any]], 
        result_type: str = "combined"
    ) -> List[Dict[str, Any]]:
        """
        LLM 결과를 표준 형식으로 변환
        
        Args:
            llm_results (List[Dict[str, Any]]): Results from LLM processing
            result_type (str): Type of results ("summary", "qa", "combined")
            
        Returns:
            List[Dict[str, Any]]: Standardized results
        """
        standardized = []
        
        for result in llm_results:
            standard_result = {
                "id": result.get("id", len(standardized)),
                "original_text": result.get("original_text", result.get("original_content", "")),
                "malware_name": result.get("malware_name", f"malware_{result.get('id', len(standardized))}"),
                "model": result.get("model", "unknown"),
                "sources_used": result.get("sources_used", [])
            }
            
            # Add type-specific fields
            if result_type in ["summary", "combined"]:
                standard_result["summary_text"] = result.get("summary", result.get("summary_text", ""))
                standard_result["prompt_version"] = result.get("prompt_version", "v2")
            
            if result_type in ["qa", "combined"]:
                standard_result["qa_text"] = result.get("qa_text", "")
                standard_result["parsed_qa"] = result.get("parsed_qa", {})
            
            standardized.append(standard_result)
        
        return standardized
    
    def create_legacy_format(
        self, 
        combined_results: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        기존 코드와 호환되는 형식으로 변환
        
        Args:
            combined_results (List[Dict[str, Any]]): Combined LLM results
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (summary_results, qa_results)
        """
        summary_results = []
        qa_results = []
        
        for result in combined_results:
            # Summary format
            if "summary_text" in result:
                summary_result = {
                    "id": result["id"],
                    "original_text": result["original_text"],
                    "summary_text": result["summary_text"]
                }
                summary_results.append(summary_result)
            
            # QA format
            if "qa_text" in result:
                qa_result = {
                    "id": result["id"],
                    "original_text": result["original_text"],
                    "qa_text": result["qa_text"]
                }
                qa_results.append(qa_result)
        
        return summary_results, qa_results
    
    def save_in_legacy_format(
        self, 
        combined_results: List[Dict[str, Any]], 
        base_filename: str = "gpt_results"
    ):
        """
        기존 형식으로 결과 저장
        
        Args:
            combined_results (List[Dict[str, Any]]): Combined results
            base_filename (str): Base filename for output
        """
        summary_results, qa_results = self.create_legacy_format(combined_results)
        
        # Save summary results (compatible with eval_summary.py)
        summary_path = f"{base_filename}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, ensure_ascii=False, indent=2)
        
        # Save QA results (compatible with analysis_qa.py)
        qa_path = f"{base_filename}_qa.json"
        with open(qa_path, 'w', encoding='utf-8') as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=2)
        
        # Save combined results
        combined_path = f"{base_filename}_combined.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved:")
        print(f"  Summary: {summary_path}")
        print(f"  QA: {qa_path}")
        print(f"  Combined: {combined_path}")
    
    def prepare_for_evaluation(
        self, 
        results: List[Dict[str, Any]], 
        evaluation_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        평가 모듈에 적합한 형식으로 데이터 준비
        
        Args:
            results (List[Dict[str, Any]]): LLM results
            evaluation_type (str): Type of evaluation ("summary" or "qa")
            
        Returns:
            Dict[str, Any]: Data prepared for evaluation
        """
        if evaluation_type == "summary":
            # Format for eval_summary.py
            eval_data = {}
            for result in results:
                doc_id = str(result.get("id", 0))
                eval_data[doc_id] = {
                    "summary": result.get("summary_text", ""),
                    "original": result.get("original_text", "")
                }
            return eval_data
        
        elif evaluation_type == "qa":
            # Format for analysis_qa.py
            eval_data = []
            for result in results:
                qa_item = {
                    "id": result.get("id", 0),
                    "qa_text": result.get("qa_text", "")
                }
                eval_data.append(qa_item)
            return eval_data
        
        else:
            raise ValueError(f"Unknown evaluation type: {evaluation_type}")


# Utility functions for legacy compatibility
def load_and_convert_scraped_data(scraped_data_path: str) -> List[str]:
    """
    스크래핑된 데이터 파일을 로드하고 텍스트 리스트로 변환
    
    Args:
        scraped_data_path (str): Path to scraped data JSON file
        
    Returns:
        List[str]: List of text content
    """
    adapter = DataAdapter()
    
    try:
        with open(scraped_data_path, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        
        return adapter.scraped_to_text_list(scraped_data)
    
    except Exception as e:
        print(f"Error loading scraped data: {e}")
        return []


def convert_legacy_text_file(text_file_path: str) -> List[str]:
    """
    기존 텍스트 파일을 변환
    
    Args:
        text_file_path (str): Path to legacy text file
        
    Returns:
        List[str]: List of articles
    """
    adapter = DataAdapter()
    return adapter.text_file_to_list(text_file_path)


if __name__ == "__main__":
    # Test the adapter
    adapter = DataAdapter()
    
    # Example scraped data
    example_scraped = [
        {
            "malware_name": "zeus",
            "combined_content": "Zeus is a banking trojan that steals credentials...",
            "metadata": {"sources_used": ["wikipedia"]}
        }
    ]
    
    # Convert to text list
    text_list = adapter.scraped_to_text_list(example_scraped)
    print(f"Converted to text list: {len(text_list)} items")
    
    # Example LLM results
    example_llm_results = [
        {
            "id": 0,
            "original_text": "Zeus is a banking trojan...",
            "summary_text": "Zeus is a malware targeting banking systems.",
            "qa_text": "{Zeus, Windows, banking vulnerability, network traffic, cybercriminal group}",
            "model": "gpt-3.5-turbo"
        }
    ]
    
    # Convert to standard format
    standardized = adapter.llm_results_to_standard_format(example_llm_results, "combined")
    print(f"Standardized: {len(standardized)} items")
    
    # Create legacy format
    summary_results, qa_results = adapter.create_legacy_format(standardized)
    print(f"Legacy format - Summary: {len(summary_results)}, QA: {len(qa_results)}")
