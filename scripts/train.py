import os
from pathlib import Path
from gensim.models import FastText, Word2Vec

from scripts.spacy_metadata import extract_metadata
from utils.model_loader import load_embedding_model

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
    # Extract metadata using spaCy before building the vocabulary
    print("Extracting metadata from the corpus using spaCy...")
    corpus_metadata = extract_metadata(sentences)

    save_path: Path = save_dir / f"{model_name}.model"
    vec_path: Path = save_dir / f"{model_name}.vec"

    if resume and save_path.exists():
        print(f"Resuming training from existing model at {save_path}")
        model = load_embedding_model(save_path, model_type)
        model.build_vocab(sentences, update=True)
    else:
        print(f"Initializing new {model_type} model...")
        if model_type.lower() == "fasttext":
            model = FastText(
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=os.cpu_count(),
                sg=1, hs=0, negative=10, sample=1e-5, seed=42,
                sorted_vocab=True, shrink_windows=True, batch_words=10000
            )
        elif model_type.lower() == "word2vec":
            model = Word2Vec(
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=os.cpu_count(),
                sg=1, hs=0, negative=10, sample=1e-5, seed=42,
                compute_loss=True, sorted_vocab=True, shrink_windows=True, batch_words=10000
            )
        else:
            raise ValueError(f"Unsupported model_type '{model_type}'")

        print("Building vocabulary from sentences...")
        model.build_vocab(sentences)

    print(f"Training {model_type} model for {epochs} epochs...")
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    # Save the model in both native and word2vec formats
    model.save(str(save_path))
    model.wv.save_word2vec_format(str(vec_path))

    training_loss = model.get_latest_training_loss()
    print(f"Final training loss: {training_loss}")
    print(f"{model_type.capitalize()} model and vectors saved to {save_path} and {vec_path}")

    # Optionally inspect the metadata for a few tokens
    vocab = model.wv.index_to_key
    for word in vocab[:5]:
        meta = corpus_metadata.get(word, {})
        print(f"Word: {word}, Metadata: {meta}")

    # Return both the model and the extracted metadata
    return model, corpus_metadata