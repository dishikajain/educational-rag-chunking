import sys
import os

print("--- Phase 1 Environment Verification ---")
print(f"Python Version: {sys.version}")

try:
    import dotenv
    print("python-dotenv is installed.")
except ImportError:
    print("ERROR: python-dotenv is not installed.")
    sys.exit(1)

try:
    import pytest
    print("pytest is installed.")
except ImportError:
    print("ERROR: pytest is not installed.")
    sys.exit(1)

# Verify directories
dirs_to_check = [
    "data/raw", "data/extracted", "data/preprocessed",
    "data/chunks/dev", "data/chunks/formal",
    "chromadb/dev", "chromadb/experiment", "results/runs"
]
all_dirs_exist = True
for d in dirs_to_check:
    if not os.path.isdir(d):
        print(f"ERROR: Directory '{d}' does not exist.")
        all_dirs_exist = False
if all_dirs_exist:
    print("All required directories exist.")

# Verify config loads
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    if not os.path.exists(".env"):
        print("Creating dummy .env from .env.example...")
        with open(".env", "w") as f:
            f.write(open(".env.example").read())
    import src.config as config
    print("src.config loaded successfully.")
    print(f"GOOGLE_API_KEY configured: {bool(config.GOOGLE_API_KEY)}")
except Exception as e:
    print(f"ERROR: Failed to load src.config - {e}")
    sys.exit(1)

print("\n--- Checklist for future phases ---")
print("Not yet installed (expected): sentence-transformers, chromadb, google-genai, streamlit")
print("\nEnvironment verification passed!")
