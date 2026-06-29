"""
GPT Client for OpenAI API interactions
"""

from openai import OpenAI
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import traceback

from config.settings import Settings


class GPTClient:
    """
    OpenAI GPT API 클라이언트
    통합된 GPT API 호출 인터페이스 제공
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = Settings.DEFAULT_MODEL,
        manifest_path: Optional[str] = None,
    ):
        """
        Initialize GPT Client

        Args:
            api_key: OpenAI API key
            model: Default logical model name (resolved to snapshot via Settings)
            manifest_path: Optional file path for per-call run manifest
                (defaults to env LLM_RUN_MANIFEST or pipeline_output/run_manifest.jsonl)
        """
        self.api_key = api_key or Settings.OPENAI_API_KEY
        self.model = model
        self.snapshot_model = Settings.resolve_model_snapshot(model)
        self.client = None
        self.manifest_path = manifest_path or os.getenv(
            "LLM_RUN_MANIFEST", "pipeline_output/run_manifest.jsonl"
        )

        print(
            f"[DEBUG] GPT 클라이언트 초기화 - API Key 존재: {bool(self.api_key)}, "
            f"모델: {self.model} (snapshot={self.snapshot_model})"
        )

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
        temperature: float = Settings.LLM_TEMPERATURE,
        top_p: float = Settings.LLM_TOP_P,
        seed: Optional[int] = Settings.LLM_SEED,
        max_tokens: Optional[int] = None,
        record_manifest: bool = True,
        **kwargs,
    ) -> str:
        """
        Send chat completion request to OpenAI.

        Args:
            messages: Chat messages
            model: Logical model to use (overrides default)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            seed: Deterministic-mode seed (OpenAI parameter)
            max_tokens: Maximum tokens to generate
            record_manifest: When True, append the run config to manifest
            **kwargs: Additional parameters forwarded to OpenAI

        Returns:
            Generated response
        """
        logical = model or self.model
        snapshot = Settings.resolve_model_snapshot(logical)

        print(
            f"[DEBUG] GPT API 호출 시작 - Model: {logical} (snapshot={snapshot}), "
            f"Temperature: {temperature}, top_p: {top_p}"
        )
        print(f"[DEBUG] 메시지 수: {len(messages)}")

        api_kwargs = dict(
            model=snapshot,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if seed is not None:
            api_kwargs["seed"] = seed
        api_kwargs.update(kwargs)

        try:
            response = self.client.chat.completions.create(**api_kwargs)
            result = response.choices[0].message.content.strip()
            print(f"[DEBUG] GPT API 응답 성공 - 응답 길이: {len(result)} 문자")

            if record_manifest:
                self._append_manifest(
                    messages=messages,
                    model_logical=logical,
                    model_snapshot=snapshot,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    response_id=getattr(response, "id", None),
                    system_fingerprint=getattr(response, "system_fingerprint", None),
                    usage=getattr(response, "usage", None),
                )
            return result

        except Exception as e:
            print(f"[ERROR] GPT API 호출 실패: {e}")
            traceback.print_exc()
            raise

    def _append_manifest(
        self,
        *,
        messages: List[Dict[str, str]],
        model_logical: str,
        model_snapshot: str,
        temperature: float,
        top_p: float,
        seed: Optional[int],
        response_id: Optional[str],
        system_fingerprint: Optional[str],
        usage: Any,
    ) -> None:
        """Append a single LLM call record to the run manifest (JSONL)."""
        try:
            Path(self.manifest_path).parent.mkdir(parents=True, exist_ok=True)
            sys_msg = next(
                (m.get("content", "") for m in messages if m.get("role") == "system"),
                "",
            )
            user_msg = "\n".join(
                m.get("content", "") for m in messages if m.get("role") == "user"
            )
            usage_dict = None
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_logical": model_logical,
                "model_snapshot": model_snapshot,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "system_msg_hash": hashlib.sha256(sys_msg.encode("utf-8")).hexdigest(),
                "user_msg_hash": hashlib.sha256(user_msg.encode("utf-8")).hexdigest(),
                "response_id": response_id,
                "system_fingerprint": system_fingerprint,
                "usage": usage_dict,
            }
            with open(self.manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] manifest 기록 실패 (무시): {e}")
    
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
