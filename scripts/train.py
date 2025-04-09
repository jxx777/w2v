from gensim.models import Word2Vec
from pathlib import Path
import os


def train_word2vec_model(
        sentences: list,
        save_dir: Path,
        *,
        model_name: str,
        vector_size: int,
        window: int,
        min_count: int,
        epochs: int
) -> Word2Vec:

    model = Word2Vec(
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

    model.build_vocab(sentences)
    print(f"Training model for {epochs} epochs on {len(sentences):,} examples...")
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    training_loss = model.get_latest_training_loss()
    print(training_loss)

    save_path = Path(save_dir) / f"{model_name}.model"
    model.save(str(save_path))

    vec_path = Path(save_dir) / f"{model_name}.vec"
    model.wv.save_word2vec_format(str(vec_path))

    print(f"Model saved to {save_path}")
    return model