#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified LLM client wrapper"""

import os
import time
import threading
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from zai import ZhipuAiClient
except ImportError:
    ZhipuAiClient = None


class LLMClient:
    """Unified LLM client, supports DeepSeek and Zhipu"""
    
    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "DeepSeek-V3.2",
        api_concurrency: int = 64,
        system_message: str = "You are a helpful assistant.",
    ):
        """
        Initialize LLM client
        
        Args:
            provider: LLM provider ("deepseek" or "zhipu")
            api_key: API key, if None will read from environment variable
            base_url: API base URL (required for DeepSeek, ignored for Zhipu)
            model_name: Model name
            api_concurrency: API concurrency limit
            system_message: System message
        """
        self.provider = (provider or "").lower().strip()
        self.system_message = system_message
        self.model_name = model_name
        self.api_semaphore = threading.BoundedSemaphore(value=api_concurrency)
        
        # Handle API key
        if not api_key:
            if self.provider == "deepseek":
                api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
            elif self.provider == "zhipu":
                api_key = os.environ.get("ZHIPU_API_KEY")
        
        if not api_key:
            raise ValueError(
                "API key missing. Provide api_key or set "
                "DEEPSEEK_API_KEY/OPENAI_API_KEY (for deepseek) or "
                "ZHIPU_API_KEY (for zhipu) in environment."
            )
        self.api_key = api_key
        
        # Handle base_url
        if self.provider == "deepseek":
            self.base_url = base_url or "https://api.deepseek.com/v1"
            if OpenAI is None:
                raise RuntimeError("openai library not found. Install via `pip install openai`.")
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.provider == "zhipu":
            self.base_url = None  # Zhipu doesn't use base_url
            if ZhipuAiClient is None:
                raise RuntimeError("zai library not found. Install Zhipu client or switch provider.")
            self.client = ZhipuAiClient(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'deepseek' or 'zhipu'.")
    
    def _extract_text(self, response) -> str:
        """Extract text content from API response"""
        try:
            if response and getattr(response, "choices", None):
                msg = response.choices[0].message
                content = getattr(msg, "content", "")
                return str(content or "").strip()
        except Exception:
            pass
        return ""
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        system_message: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Generate text
        
        Args:
            prompt: User prompt
            temperature: Temperature parameter
            top_p: top_p parameter
            max_tokens: Maximum token count
            system_message: System message (if None, uses the one from initialization)
            max_retries: Maximum retry count
            
        Returns:
            Generated text content
        """
        system_msg = system_message or self.system_message
        
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                
                with self.api_semaphore:
                    resp = self.client.chat.completions.create(**kwargs)
                
                text = self._extract_text(resp)
                if text:
                    return text
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1.5 ** attempt)  # Exponential backoff
                else:
                    raise
        return ""
