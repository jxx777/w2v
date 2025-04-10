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
        epochs: int,
        resume: bool = False
):
    save_path = save_dir / f"{model_name}.model"
    vec_path = save_dir / f"{model_name}.vec"

    if resume and save_path.exists():
        print(f"Resuming training from existing model at {save_path}")
        match model_type.lower():
            case "fasttext":
                model = FastText.load(str(save_path))
            case "word2vec":
                model = Word2Vec.load(str(save_path))
            case _:
                raise ValueError(f"Unsupported model_type '{model_type}'.")
        model.build_vocab(sentences, update=True)
    else:
        print(f"Initializing new {model_type} model...")
        match model_type.lower():
            case "fasttext":
                model = FastText(
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
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
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
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
                raise ValueError(f"Unsupported model_type '{model_type}'.")

        print(f"Building vocabulary....")
        model.build_vocab(sentences)

    print(f"Training {model_type} model for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train(
            sentences,
            total_examples=model.corpus_count,
            epochs=1
        )

        # Save checkpoint after each epoch
        model.save(str(save_path)) # Saves to .model
        model.wv.save_word2vec_format(str(vec_path)) # Saves to .vec
        print(f"Checkpoint saved after epoch {epoch + 1}")

    training_loss = model.get_latest_training_loss()
    print(f"Final training loss: {training_loss}")
    print(f"{model_type.capitalize()} model and vectors saved to {save_path} and {vec_path}")
    return model