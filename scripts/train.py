from gensim.models import FastText, Word2Vec
from pathlib import Path
import os


def train_embedding_model(
        sentences,
        save_dir: Path,
        *,
        model_type: str,
        model_name: str,
        vector_size: int,
        window: int,
        min_count: int,
        epochs: int
):
    match model_type.lower():
        case "fasttext":
            model = FastText(
                sentences=sentences,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                epochs=epochs,
                workers=os.cpu_count(),
                sg=1,
                hs=0,
                negative=10,
                sample=1e-5,
                seed=42,
                sorted_vocab=True,
                shrink_windows=True,
                batch_words=10000
            )
        case "word2vec":
            model = Word2Vec(
                sentences=sentences,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                epochs=epochs,
                workers=os.cpu_count(),
                sg=1,
                hs=0,
                negative=10,
                sample=1e-5,
                seed=42,
                compute_loss=True,
                sorted_vocab=True,
                shrink_windows=True,
                batch_words=10000
            )
        case _:
            raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'fasttext' or 'word2vec'.")

    print(f"{model_type.capitalize()} training completed for {epochs} epochs on {model.corpus_count:,} documents.")
    training_loss = model.get_latest_training_loss()
    print(f"Training loss: {training_loss}")

    save_path: Path = save_dir / f"{model_name}.model"
    model.save(str(save_path))

    # Docs: Store the input-hidden weight matrix in the same format used by the original C word2vec-tool, for compatibility.
    vec_path: Path = save_dir / f"{model_name}.vec"
    model.wv.save_word2vec_format(str(vec_path))

    print(f"{model_type.capitalize()} model and vectors saved to {save_path} and {vec_path}")
    return model