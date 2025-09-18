"""
Question Answering module for CTI domain
"""

from typing import List, Dict, Any, Optional, Tuple
import json
from tqdm import tqdm
import time
import traceback

from .gpt_client import GPTClient
from config.settings import Settings


class QuestionAnsweringHandler:
    """
    CTI 도메인 Question Answering 처리 클래스
    5개 카테고리(Malware, System, Indicator, Vulnerability, Organization)에 대한 질답 수행
    """
    
    def __init__(self, 
                 model: str = Settings.DEFAULT_MODEL,
                 api_key: Optional[str] = None):
        """
        Initialize QA Handler
        
        Args:
            model (str): Model to use for QA
            api_key (Optional[str]): OpenAI API key
        """
        self.model = model
        self.client = GPTClient(api_key, model)
        self.labels = Settings.QA_LABELS
    
    def create_qa_prompt(self, source_text: str) -> str:
        """
        Create QA prompt for CTI domain
        
        Args:
            source_text (str): Input text for QA
            
        Returns:
            str: Formatted QA prompt
        """
        prompt = f"""
        # Main Role
        - You're a famous NER Expert in CTI (Cyber Threat Intelligence) domain. Your task is to generate several questions that ask meaningful information about the content of a malware article retrieved from a popular malware blog post.

        # Main Instructions
        - Answer questions from blog text data given as input. The questions are also specified below. Answer each question in a section enclosed by three backticks.
        - Please note that these blogs are all related to CTI.
        - All answers to the questions must be short answers. It would be great if you could answer the question with just one entity.
        - If you determine that there is no appropriate answer to a question, you MUST print 'no answer'.
        - You only need to provide data on the answers to five questions, separated by commas.
        
        ## Input blog text data
        {source_text}
        
        # Questions
        Question 1 : What is the main MALWARE covered in this article?
        Answer 1 : ```Your Answer for Question 1```
        Question 2 : What is the main SYSTEM covered in this article?
        Answer 2 : ```Your Answer for Question 2```
        Question 3 : What is the main VULNERABILITY covered in this article?
        Answer 3 : ```Your Answer for Question 3```
        Question 4 : What is the main INDICATOR covered in this article?
        Answer 4 : ```Your Answer for Question 4```
        Question 5 : What is the main ORGANIZATION covered in this article?
        Answer 5 : ```Your Answer for Question 5```
        
        # You Expected Responses
        - Remove triple backticks when you response
        {{Answer1, Answer2, Answer3, Answer4, Answer5}}
        """
        return prompt
    
    def qa_article(self, source_text: str) -> str:
        """
        단일 문서에 대한 QA 수행
        
        Args:
            source_text (str): Input text for QA
            
        Returns:
            str: QA response in format {Answer1, Answer2, Answer3, Answer4, Answer5}
        """
        try:
            print(f"[DEBUG] QA 시작 - 텍스트 길이: {len(source_text)} 문자")
            
            prompt = self.create_qa_prompt(source_text)
            print(f"[DEBUG] QA 프롬프트 생성 완료 - 프롬프트 길이: {len(prompt)} 문자")
            
            print("[DEBUG] GPT 클라이언트에 QA 요청 전송 중...")
            response = self.client.simple_prompt(
                prompt,
                temperature=Settings.QA_TEMPERATURE
            )
            
            print(f"[DEBUG] QA 완료 - 결과 길이: {len(response)} 문자")
            return response
            
        except Exception as e:
            print(f"[ERROR] QA 처리 중 오류 발생: {e}")
            traceback.print_exc()
            raise
    
    def parse_qa_response(self, qa_response: str) -> Dict[str, str]:
        """
        Parse QA response into structured format
        
        Args:
            qa_response (str): Raw QA response
            
        Returns:
            Dict[str, str]: Parsed QA responses mapped to labels
        """
        try:
            # Remove curly braces and split by comma
            clean_response = qa_response.strip("{} ")
            answers = [answer.strip() for answer in clean_response.split(',')]
            
            # Map answers to labels
            if len(answers) == len(self.labels):
                return dict(zip(self.labels, answers))
            else:
                print(f"Warning: Expected {len(self.labels)} answers, got {len(answers)}")
                # Pad with "no answer" if needed
                while len(answers) < len(self.labels):
                    answers.append("no answer")
                return dict(zip(self.labels, answers[:len(self.labels)]))
                
        except Exception as e:
            print(f"Error parsing QA response: {e}")
            # Return default structure
            return {label: "no answer" for label in self.labels}
    
    def batch_qa(
        self, 
        texts: List[str], 
        save_path: Optional[str] = None,
        start_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        배치 QA 처리
        
        Args:
            texts (List[str]): List of texts for QA
            save_path (Optional[str]): Path to save intermediate results
            start_idx (int): Starting index for processing
            
        Returns:
            List[Dict[str, Any]]: List of QA results
        """
        results = []
        err_results = []
        
        for i in tqdm(range(start_idx, len(texts)), desc="Processing QA"):
            text = texts[i].strip()
            
            try:
                qa_response = self.qa_article(text)
                parsed_qa = self.parse_qa_response(qa_response)
                
                result = {
                    "id": i,
                    "original_text": text,
                    "qa_text": qa_response,
                    "parsed_qa": parsed_qa,
                    "model": self.model
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
    
    def qa_scraped_data(
        self, 
        scraped_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        스크래핑된 데이터에 대한 QA 처리
        
        Args:
            scraped_data (List[Dict[str, Any]]): Data from data scraper
            
        Returns:
            List[Dict[str, Any]]: QA results
        """
        results = []
        
        for data in tqdm(scraped_data, desc="Processing scraped data QA"):
            try:
                # Extract combined content for QA
                content = data.get("combined_content", "")
                
                if not content.strip():
                    print(f"No content found for {data.get('malware_name', 'unknown')}")
                    continue
                
                qa_response = self.qa_article(content)
                parsed_qa = self.parse_qa_response(qa_response)
                
                result = {
                    "malware_name": data.get("malware_name"),
                    "original_content": content,
                    "qa_text": qa_response,
                    "parsed_qa": parsed_qa,
                    "model": self.model,
                    "sources_used": data.get("metadata", {}).get("sources_used", [])
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing QA for {data.get('malware_name', 'unknown')}: {e}")
                continue
        
        return results
    
    def combined_summarization_qa(
        self,
        texts: List[str],
        summarizer,  # AbstractiveSummarizer instance
        save_path: Optional[str] = None,
        start_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Combined processing for both summarization and QA
        
        Args:
            texts (List[str]): List of texts to process
            summarizer: AbstractiveSummarizer instance
            save_path (Optional[str]): Path to save results
            start_idx (int): Starting index
            
        Returns:
            List[Dict[str, Any]]: Combined results
        """
        results = []
        err_results = []
        
        for i in tqdm(range(start_idx, len(texts)), desc="Processing Summary+QA"):
            text = texts[i].strip()
            
            try:
                # Generate summary
                summary = summarizer.summarize_article(text)
                
                # Generate QA
                qa_response = self.qa_article(text)
                parsed_qa = self.parse_qa_response(qa_response)
                
                result = {
                    "id": i,
                    "original_text": text,
                    "summary_text": summary,
                    "qa_text": qa_response,
                    "parsed_qa": parsed_qa,
                    "model": self.model
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
                
                time.sleep(3)
            
            # Periodic backup
            if save_path and ((i + 1) % 100 == 0 or i == len(texts) - 1):
                self._save_intermediate_results(results, save_path)
                print(f"[{i + 1}] 중간 저장 완료.")
        
        return results
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]], save_path: str):
        """중간 결과 저장"""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def get_labels(self) -> List[str]:
        """Get the list of QA labels"""
        return self.labels.copy()


# Legacy function for backward compatibility
def qa_article(source_text: str, model: str = Settings.DEFAULT_MODEL) -> str:
    """
    Legacy wrapper function for QA
    
    Args:
        source_text (str): Input text for QA
        model (str): Model to use
        
    Returns:
        str: QA response
    """
    qa_handler = QuestionAnsweringHandler(model)
    return qa_handler.qa_article(source_text)


if __name__ == "__main__":
    # Test the QA handler
    if Settings.validate_api_key():
        qa_handler = QuestionAnsweringHandler()
        
        test_text = """
        Zeus is a Trojan horse malware package that runs on versions of Microsoft Windows. 
        It targets banking information and uses man-in-the-browser attacks. 
        The malware affects Windows systems and is associated with the Zeus criminal organization.
        It exploits browser vulnerabilities and uses network indicators for communication.
        """
        
        qa_response = qa_handler.qa_article(test_text)
        parsed_qa = qa_handler.parse_qa_response(qa_response)
        
        print(f"QA Response: {qa_response}")
        print(f"Parsed QA: {parsed_qa}")
    else:
        print("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
