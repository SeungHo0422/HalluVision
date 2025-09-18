"""
Abstractive Summarization module using LLM
"""

from typing import List, Dict, Any, Optional
import json
from tqdm import tqdm
import time
import traceback

from .gpt_client import GPTClient
from config.settings import Settings


class AbstractiveSummarizer:
    """
    추상적 요약(Abstractive Summarization) 처리 클래스
    CTI 도메인에 특화된 악성코드 정보 요약 기능 제공
    """
    
    def __init__(self, 
                 model: str = Settings.DEFAULT_MODEL,
                 api_key: Optional[str] = None):
        """
        Initialize Abstractive Summarizer
        
        Args:
            model (str): Model to use for summarization
            api_key (Optional[str]): OpenAI API key
        """
        self.model = model
        self.client = GPTClient(api_key, model)
        
        # Prompt versions
        self.prompt_versions = {
            "v1": self._prompt_ver1,
            "v2": self._prompt_ver2
        }
        
        self.current_version = "v2"  # Default to version 2
    
    def _prompt_ver1(self, source_text: str) -> str:
        """
        기존 프롬프트 버전. instruction에 문서 내 정보만을 이용하여 요약하라고 지시
        
        Args:
            source_text (str): Input text to summarize
            
        Returns:
            str: Formatted prompt
        """
        prompt = f"""
        # Main Role
        - You're a famous NER Expert in CTI (Cyber Threat Intelligence) domain. Your task is to generate a short summary of a malware information obtained from popular malware blog posts, following the instructions below.
        
        # Main Instructions
        - Summarize the content below in at least {Settings.SUMMARY_MIN_WORDS} words, and at most {Settings.SUMMARY_MAX_WORDS} words.
        - You should answer using only the words and contents in the summary as much as possible.
        
        ## Input blog text data
        {source_text}
        """
        return prompt
    
    def _prompt_ver2(self, source_text: str) -> str:
        """
        개선된 프롬프트 버전. 문서 제약 조건 제거
        
        Args:
            source_text (str): Input text to summarize
            
        Returns:
            str: Formatted prompt
        """
        prompt = f"""
        # Main Role
        - You're a famous Expert in CTI (Cyber Threat Intelligence) domain. Your task is to generate a short summary of a malware information obtained from popular malware blog posts, following the instructions below.
    
        # Main Instructions
        - Summarize the content below in at least {Settings.SUMMARY_MIN_WORDS} words, and at most {Settings.SUMMARY_MAX_WORDS} words.
    
        ## Input blog text data
        {source_text}
        """
        return prompt
    
    def summarize_article(self, source_text: str, prompt_version: str = None) -> str:
        """
        단일 문서 요약
        
        Args:
            source_text (str): Input text to summarize
            prompt_version (str): Prompt version to use ("v1" or "v2")
            
        Returns:
            str: Generated summary
        """
        try:
            print(f"[DEBUG] 요약 시작 - 텍스트 길이: {len(source_text)} 문자")
            
            version = prompt_version or self.current_version
            print(f"[DEBUG] 프롬프트 버전: {version}")
            
            if version not in self.prompt_versions:
                raise ValueError(f"Unknown prompt version: {version}")
            
            prompt = self.prompt_versions[version](source_text)
            print(f"[DEBUG] 프롬프트 생성 완료 - 프롬프트 길이: {len(prompt)} 문자")
            
            print("[DEBUG] GPT 클라이언트에 요약 요청 전송 중...")
            response = self.client.simple_prompt(
                prompt,
                temperature=Settings.SUMMARY_TEMPERATURE
            )
            
            print(f"[DEBUG] 요약 완료 - 결과 길이: {len(response)} 문자")
            return response
            
        except Exception as e:
            print(f"[ERROR] 요약 처리 중 오류 발생: {e}")
            traceback.print_exc()
            raise
    
    def batch_summarize(
        self, 
        texts: List[str], 
        prompt_version: str = None,
        save_path: Optional[str] = None,
        start_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        배치 요약 처리
        
        Args:
            texts (List[str]): List of texts to summarize
            prompt_version (str): Prompt version to use
            save_path (Optional[str]): Path to save intermediate results
            start_idx (int): Starting index for processing
            
        Returns:
            List[Dict[str, Any]]: List of summarization results
        """
        results = []
        err_results = []
        
        for i in tqdm(range(start_idx, len(texts)), desc="Summarizing"):
            text = texts[i].strip()
            
            try:
                summary = self.summarize_article(text, prompt_version)
                
                result = {
                    "id": i,
                    "original_text": text,
                    "summary_text": summary,
                    "model": self.model,
                    "prompt_version": prompt_version or self.current_version
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"[{i}] 예외 발생: {e}")
                traceback.print_exc()
                
                error_result = {
                    "id": i,
                    "original_text": text,
                    "error": str(e)
                }
                err_results.append(error_result)
                
                time.sleep(3)  # Error recovery delay
            
            # Periodic backup
            if save_path and ((i + 1) % 100 == 0 or i == len(texts) - 1):
                self._save_intermediate_results(results, save_path)
                print(f"[{i + 1}] 중간 저장 완료.")
        
        # Save error results if any
        if err_results and save_path:
            error_path = save_path.replace('.json', '_errors.json')
            self._save_intermediate_results(err_results, error_path)
        
        return results
    
    def summarize_scraped_data(
        self, 
        scraped_data: List[Dict[str, Any]],
        prompt_version: str = None
    ) -> List[Dict[str, Any]]:
        """
        스크래핑된 데이터에 대한 요약 처리
        
        Args:
            scraped_data (List[Dict[str, Any]]): Data from data scraper
            prompt_version (str): Prompt version to use
            
        Returns:
            List[Dict[str, Any]]: Summarized data
        """
        results = []
        
        for data in tqdm(scraped_data, desc="Processing scraped data"):
            try:
                # Extract combined content for summarization
                content = data.get("combined_content", "")
                
                if not content.strip():
                    print(f"No content found for {data.get('malware_name', 'unknown')}")
                    continue
                
                summary = self.summarize_article(content, prompt_version)
                
                result = {
                    "malware_name": data.get("malware_name"),
                    "original_content": content,
                    "summary": summary,
                    "model": self.model,
                    "prompt_version": prompt_version or self.current_version,
                    "sources_used": data.get("metadata", {}).get("sources_used", [])
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {data.get('malware_name', 'unknown')}: {e}")
                continue
        
        return results
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]], save_path: str):
        """중간 결과 저장"""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def set_prompt_version(self, version: str):
        """
        Set the prompt version to use
        
        Args:
            version (str): Prompt version ("v1" or "v2")
        """
        if version not in self.prompt_versions:
            raise ValueError(f"Unknown prompt version: {version}")
        
        self.current_version = version
    
    def get_available_versions(self) -> List[str]:
        """Get list of available prompt versions"""
        return list(self.prompt_versions.keys())


# Legacy function for backward compatibility
def summarize_article(source_text: str, model: str = Settings.DEFAULT_MODEL) -> str:
    """
    Legacy wrapper function for summarization
    
    Args:
        source_text (str): Input text to summarize
        model (str): Model to use
        
    Returns:
        str: Generated summary
    """
    summarizer = AbstractiveSummarizer(model)
    return summarizer.summarize_article(source_text)


if __name__ == "__main__":
    # Test the summarizer
    if Settings.validate_api_key():
        summarizer = AbstractiveSummarizer()
        
        test_text = """
        Zeus is a Trojan horse malware package that runs on versions of Microsoft Windows. 
        While it can be used to carry out many malicious and criminal tasks, it is often used to steal banking information by man-in-the-browser keystroke logging and form grabbing. 
        It is also used to install the CryptoLocker ransomware. Zeus is spread mainly through drive-by downloads and phishing schemes.
        """
        
        summary = summarizer.summarize_article(test_text)
        print(f"Original: {test_text}")
        print(f"Summary: {summary}")
    else:
        print("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
