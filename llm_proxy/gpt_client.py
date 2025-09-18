"""
GPT Client for OpenAI API interactions
"""

from openai import OpenAI
import os
from typing import Dict, Any, Optional, List
import time
import traceback

from config.settings import Settings


class GPTClient:
    """
    OpenAI GPT API 클라이언트
    통합된 GPT API 호출 인터페이스 제공
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = Settings.DEFAULT_MODEL):
        """
        Initialize GPT Client
        
        Args:
            api_key (Optional[str]): OpenAI API key
            model (str): Default model to use
        """
        self.api_key = api_key or Settings.OPENAI_API_KEY
        self.model = model
        self.client = None
        
        print(f"[DEBUG] GPT 클라이언트 초기화 - API Key 존재: {bool(self.api_key)}, 모델: {self.model}")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            print("[DEBUG] OpenAI 클라이언트 초기화 시작...")
            # Set OpenAI API key (for compatibility with older versions)
            OpenAI.api_key = self.api_key
            self.client = OpenAI(api_key=self.api_key)
            print("[DEBUG] OpenAI 클라이언트 초기화 완료")
        except Exception as e:
            print(f"[ERROR] OpenAI 클라이언트 초기화 실패: {e}")
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send chat completion request to OpenAI
        
        Args:
            messages (List[Dict[str, str]]): Chat messages
            model (Optional[str]): Model to use (overrides default)
            temperature (float): Sampling temperature
            max_tokens (Optional[int]): Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        print(f"[DEBUG] GPT API 호출 시작 - Model: {model or self.model}, Temperature: {temperature}")
        print(f"[DEBUG] 메시지 수: {len(messages)}")
        
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            result = response.choices[0].message.content.strip()
            print(f"[DEBUG] GPT API 응답 성공 - 응답 길이: {len(result)} 문자")
            return result
            
        except Exception as e:
            print(f"[ERROR] GPT API 호출 실패: {e}")
            traceback.print_exc()
            raise
    
    def simple_prompt(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0,
        **kwargs
    ) -> str:
        """
        Send simple prompt to GPT
        
        Args:
            prompt (str): Input prompt
            model (Optional[str]): Model to use
            temperature (float): Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, model, temperature, **kwargs)
    
    def batch_processing(
        self, 
        prompts: List[str], 
        model: Optional[str] = None,
        temperature: float = 0,
        delay: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        Process multiple prompts with rate limiting
        
        Args:
            prompts (List[str]): List of prompts to process
            model (Optional[str]): Model to use
            temperature (float): Sampling temperature
            delay (float): Delay between requests (seconds)
            **kwargs: Additional parameters
            
        Returns:
            List[str]: List of responses
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                response = self.simple_prompt(prompt, model, temperature, **kwargs)
                results.append(response)
                
                # Rate limiting
                if i < len(prompts) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Error processing prompt {i}: {e}")
                results.append(f"Error: {str(e)}")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "model": self.model,
            "api_key_set": bool(self.api_key),
            "client_initialized": self.client is not None
        }


# Legacy function for backward compatibility
def get_openai_client(api_key: Optional[str] = None) -> GPTClient:
    """
    Legacy wrapper to get GPT client
    
    Args:
        api_key (Optional[str]): OpenAI API key
        
    Returns:
        GPTClient: Initialized GPT client
    """
    return GPTClient(api_key)


if __name__ == "__main__":
    # Test the client
    if Settings.validate_api_key():
        client = GPTClient()
        
        test_prompt = "What is machine learning?"
        response = client.simple_prompt(test_prompt)
        
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response[:100]}...")
    else:
        print("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
