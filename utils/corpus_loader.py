import os
import pickle
from pathlib import Path
from gensim.corpora import WikiCorpus
from tqdm import tqdm


def load_or_tokenize_wiki(dataset_path: Path, checkpoint_path: Path):
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f) # Checkpointed sentences are returned

    wiki = WikiCorpus(
        fname=str(dataset_path),
        processes=os.cpu_count(),
        article_min_tokens=15,
        token_min_len=2,
        token_max_len=30,
        lower=True
    )

    sentences = list(tqdm(wiki.get_texts(), desc="Forming sentences from Wikipedia articles"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(sentences, f)
    return sentences
