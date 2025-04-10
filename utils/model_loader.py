from pathlib import Path
from gensim.models import FastText, Word2Vec

def load_embedding_model(model_path: Path, model_type: str):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist at {model_path}")

    match model_type.lower():
        case "fasttext":
            return FastText.load(str(model_path))
        case "word2vec":
            return Word2Vec.load(str(model_path))
        case _:
            raise ValueError(f"Unsupported model_type '{model_type}'")