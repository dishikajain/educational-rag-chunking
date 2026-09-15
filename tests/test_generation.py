import pytest
import os
import copy
from unittest.mock import patch, MagicMock

from src.generation.config import GenerationConfig, load_config
from src.generation.generator import GeminiGenerator
from src.retrieval.models import RetrievedChunk
from src.generation.models import GenerationResult

@pytest.fixture
def valid_config_dict():
    return {
        "model_name": "gemini-2.5-flash",
        "temperature": 0.0,
        "thinking_budget": 0,
        "max_output_tokens": 1024,
        "system_instruction": "Test instruction"
    }

@pytest.fixture
def mock_config(valid_config_dict):
    return GenerationConfig(**valid_config_dict)

@pytest.fixture
def sample_chunks():
    return [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            strategy="baseline_fixed_size",
            text="Text chunk 1",
            distance=0.1,
            char_start=0,
            char_end=12,
            provenance_spans=[{"char_start": 0, "char_end": 12, "label": "P"}],
            metadata={"source": "test"}
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="d1",
            strategy="baseline_fixed_size",
            text="Text chunk 2",
            distance=0.2,
            char_start=13,
            char_end=25,
            provenance_spans=[{"char_start": 13, "char_end": 25, "label": "P"}],
            metadata={}
        )
    ]

class TestConfiguration:
    def test_invalid_model(self, valid_config_dict):
        valid_config_dict["model_name"] = "gemini-1.5-flash"
        with pytest.raises(ValueError):
            GenerationConfig(**valid_config_dict)

    def test_invalid_temperature(self, valid_config_dict):
        valid_config_dict["temperature"] = 0.5
        with pytest.raises(ValueError):
            GenerationConfig(**valid_config_dict)

    def test_invalid_thinking_budget(self, valid_config_dict):
        valid_config_dict["thinking_budget"] = 1
        with pytest.raises(ValueError):
            GenerationConfig(**valid_config_dict)

    def test_invalid_max_output_tokens(self, valid_config_dict):
        valid_config_dict["max_output_tokens"] = 2048
        with pytest.raises(ValueError):
            GenerationConfig(**valid_config_dict)

    def test_missing_required_fields(self, valid_config_dict):
        del valid_config_dict["temperature"]
        with pytest.raises(ValueError):
            GenerationConfig(**valid_config_dict)

    def test_unknown_fields_rejected(self, valid_config_dict):
        valid_config_dict["unknown_field"] = "bad"
        with pytest.raises(ValueError):
            GenerationConfig(**valid_config_dict)

    def test_missing_yaml_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("does_not_exist.yaml")


class TestAPIKeyValidation:
    def test_missing_api_key(self, mock_config, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is missing or empty"):
            GeminiGenerator(mock_config)

    def test_empty_api_key(self, mock_config, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "")
        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is missing or empty"):
            GeminiGenerator(mock_config)

    def test_whitespace_api_key(self, mock_config, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "   ")
        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is missing or empty"):
            GeminiGenerator(mock_config)


class TestPromptConstructionAndValidation:
    @patch("src.generation.generator.genai.Client")
    def test_prompt_construction(self, mock_client, mock_config, sample_chunks, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
        generator = GeminiGenerator(mock_config)
        
        prompt = generator._construct_prompt("What is it?", sample_chunks)
        assert "--- START CHUNK c1 ---\nText chunk 1\n--- END CHUNK c1 ---" in prompt
        assert "--- START CHUNK c2 ---\nText chunk 2\n--- END CHUNK c2 ---" in prompt
        assert prompt.find("c1") < prompt.find("c2")
        assert "QUESTION:\nWhat is it?" in prompt

    @patch("src.generation.generator.genai.Client")
    def test_input_validation(self, mock_client, mock_config, sample_chunks, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
        generator = GeminiGenerator(mock_config)
        
        with pytest.raises(ValueError, match="Question must be a non-empty string"):
            generator.generate("   ", sample_chunks)
            
        with pytest.raises(ValueError, match="chunks must be a list"):
            generator.generate("Valid?", "not a list")
            
        with pytest.raises(ValueError, match="chunks must contain only RetrievedChunk objects"):
            generator.generate("Valid?", ["not a chunk"])


class TestGenerator:
    @patch("src.generation.generator.genai.Client")
    def test_successful_generation(self, mock_client_class, mock_config, sample_chunks, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a mocked answer."
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        # Deepcopy to verify immutability
        chunks_copy = copy.deepcopy(sample_chunks)
        
        generator = GeminiGenerator(mock_config)
        result = generator.generate("What is it?", sample_chunks)
        
        # Verify mocked API request
        mock_client.models.generate_content.assert_called_once()
        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs['model'] == "gemini-2.5-flash"
        gen_config = call_args.kwargs['config']
        assert gen_config.temperature == 0.0
        assert gen_config.max_output_tokens == 1024
        assert gen_config.system_instruction == mock_config.system_instruction
        assert gen_config.thinking_config.thinking_budget == 0
        
        # Verify GenerationResult
        assert isinstance(result, GenerationResult)
        assert result.answer == "This is a mocked answer."
        assert result.original_question == "What is it?"
        assert result.model_identifier == "gemini-2.5-flash"
        assert result.prompt_used == call_args.kwargs['contents']
        assert result.metadata["temperature"] == 0.0
        assert result.metadata["thinking_budget"] == 0
        assert result.metadata["max_output_tokens"] == 1024
        assert result.metadata["system_instruction"] == mock_config.system_instruction
        
        assert len(result.retrieved_chunks) == 2
        assert result.retrieved_chunks[0].chunk_id == "c1"
        assert result.retrieved_chunks[1].chunk_id == "c2"
        
        # Verify Immutability
        assert sample_chunks == chunks_copy

    @patch("src.generation.generator.genai.Client")
    def test_api_failure_handling(self, mock_client_class, mock_config, sample_chunks, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
        
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Timeout")
        mock_client_class.return_value = mock_client
        
        generator = GeminiGenerator(mock_config)
        
        with pytest.raises(RuntimeError, match="Gemini API generation failed: API Timeout"):
            generator.generate("What is it?", sample_chunks)
            
    @patch("src.generation.generator.genai.Client")
    def test_empty_response(self, mock_client_class, mock_config, sample_chunks, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        generator = GeminiGenerator(mock_config)
        
        with pytest.raises(RuntimeError, match="Gemini returned an empty response"):
            generator.generate("What is it?", sample_chunks)
