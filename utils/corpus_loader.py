import os
import pickle

from gensim.corpora import WikiCorpus
from pathlib import Path
from tqdm import tqdm

from corpora_checkpoints.SerializedCorpus import SerializedCorpus
from corpora_checkpoints.StreamingCorpus import StreamingCorpus


def load_or_tokenize_wiki(
        dataset_path: Path,
        checkpoint_path: Path,
        use_streaming: bool = True
):
    if use_streaming:
        # Streaming approach: use a text file.
        if checkpoint_path.exists():
            return StreamingCorpus(checkpoint_path)
        else:
            wiki = WikiCorpus(
                fname=str(dataset_path),
                processes=os.cpu_count(),
                article_min_tokens=15,
                token_min_len=2,
                token_max_len=30,
                lower=True
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for tokens in tqdm(wiki.get_texts(), desc="Processing Wikipedia articles (streaming)"):
                    f.write(" ".join(tokens) + "\n")
            return StreamingCorpus(checkpoint_path)
    else:
        # Serialized approach: use a pickle file.
        if checkpoint_path.exists():
            return SerializedCorpus(checkpoint_path)
        else:
            wiki = WikiCorpus(
                fname=str(dataset_path),
                processes=os.cpu_count(),
                article_min_tokens=15,
                token_min_len=2,
                token_max_len=30,
                lower=True
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            sentences = list(tqdm(wiki.get_texts(), desc="Processing Wikipedia articles (serialized)"))
            with open(checkpoint_path, "wb") as f:
                pickle.dump(sentences, f)
            return SerializedCorpus(checkpoint_path)
