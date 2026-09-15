import os
from typing import List
from google import genai
from google.genai import types

from src.retrieval.models import RetrievedChunk
from src.generation.config import GenerationConfig
from src.generation.models import GenerationResult

class GeminiGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or not api_key.strip():
            raise ValueError("GEMINI_API_KEY environment variable is missing or empty")
            
        self.client = genai.Client()
        
    def _construct_prompt(self, question: str, chunks: List[RetrievedChunk]) -> str:
        prompt_parts = ["CONTEXT:\n"]
        for chunk in chunks:
            prompt_parts.append(f"--- START CHUNK {chunk.chunk_id} ---\n{chunk.text}\n--- END CHUNK {chunk.chunk_id} ---\n")
        prompt_parts.append(f"\nQUESTION:\n{question}\n")
        return "".join(prompt_parts)

    def generate(self, question: str, chunks: List[RetrievedChunk]) -> GenerationResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")
            
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list")
        for chunk in chunks:
            if not isinstance(chunk, RetrievedChunk):
                raise ValueError("chunks must contain only RetrievedChunk objects")
            
        prompt = self._construct_prompt(question, chunks)
        
        # Setup generation config with thinking budget
        gen_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            system_instruction=self.config.system_instruction,
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.config.thinking_budget
            )
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=gen_config
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API generation failed: {str(e)}") from e
            
        if not response.text:
            raise RuntimeError("Gemini returned an empty response")
            
        metadata = {
            "temperature": self.config.temperature,
            "thinking_budget": self.config.thinking_budget,
            "max_output_tokens": self.config.max_output_tokens,
            "system_instruction": self.config.system_instruction
        }
            
        return GenerationResult(
            original_question=question,
            answer=response.text,
            model_identifier=self.config.model_name,
            prompt_used=prompt,
            retrieved_chunks=chunks,
            metadata=metadata
        )
