from src.retrieval.retriever import Retriever
from src.generation.generator import GeminiGenerator
from src.pipeline.models import RAGResult
from src.pipeline.exceptions import PipelineError

class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: GeminiGenerator):
        self.retriever = retriever
        self.generator = generator

    def run(self, question: str, strategy: str) -> RAGResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy must be a non-empty string")

        try:
            chunks = self.retriever.retrieve(question, strategy)
        except (ValueError, RuntimeError) as e:
            raise PipelineError(f"Retrieval failed: {str(e)}") from e
        
        try:
            generation_result = self.generator.generate(question, chunks)
        except (ValueError, RuntimeError) as e:
            raise PipelineError(f"Generation failed: {str(e)}") from e

        return RAGResult(
            question=question,
            strategy=strategy,
            generation=generation_result
        )
