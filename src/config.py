import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Required environment variable {name} is missing.")
    return value

GOOGLE_API_KEY = get_env_var("GOOGLE_API_KEY")
CHROMA_PERSIST_PATH = get_env_var("CHROMA_PERSIST_PATH")
CHROMA_DEV_PATH = get_env_var("CHROMA_DEV_PATH")
CHROMA_EXPERIMENT_PATH = get_env_var("CHROMA_EXPERIMENT_PATH")
DATA_RAW_PATH = get_env_var("DATA_RAW_PATH")
DATA_EXTRACTED_PATH = get_env_var("DATA_EXTRACTED_PATH")
DATA_PREPROCESSED_PATH = get_env_var("DATA_PREPROCESSED_PATH")
DATA_CHUNKS_DEV_PATH = get_env_var("DATA_CHUNKS_DEV_PATH")
DATA_CHUNKS_FORMAL_PATH = get_env_var("DATA_CHUNKS_FORMAL_PATH")
RESULTS_PATH = get_env_var("RESULTS_PATH")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
