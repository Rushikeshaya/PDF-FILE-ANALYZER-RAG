from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = Path("model") / "all-MiniLM-L6-v2"

def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.is_file():
        MODEL_PATH.unlink()

    print("Downloading the embedding model...")
    print("This requires an internet connection the first time.")

    model = SentenceTransformer(MODEL_NAME)
    model.save(str(MODEL_PATH))

    print("\nModel downloaded and saved successfully!")
    print(f"Model location: {MODEL_PATH.resolve()}")

    test_embedding = model.encode("This is a test sentence.")
    print(f"Embedding dimensions: {len(test_embedding)}")

if __name__ == "__main__":
    main()