import os
import pickle
from pathlib import Path
from tqdm import tqdm
from gensim.corpora import WikiCorpus

from corpora_checkpoints.SerializedCorpus import SerializedCorpus
from corpora_checkpoints.StreamingCorpus import StreamingCorpus

def get_corpus(checkpoint_path: Path, use_streaming: bool):
    match use_streaming:
        case True:
            return StreamingCorpus(checkpoint_path)
        case False:
            return SerializedCorpus(checkpoint_path)

def load_or_tokenize_wiki(
        dataset_path: Path,
        checkpoint_path: Path,
        use_streaming: bool = True
):
    if checkpoint_path.exists():
        return get_corpus(checkpoint_path, use_streaming)

    # Create a WikiCorpus instance using standardized parameters.
    wiki = WikiCorpus(
        fname=str(dataset_path),
        processes=os.cpu_count(),
        article_min_tokens=15,
        token_min_len=2,
        token_max_len=30,
        lower=True
    )

    # Set mode based on the chosen strategy.
    mode = "streaming" if use_streaming else "serialized"
    process_wiki_articles(wiki, checkpoint_path, mode)

    return get_corpus(checkpoint_path, use_streaming)

def process_wiki_articles(wiki: WikiCorpus, checkpoint_path: Path, mode: str) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    match mode:
        case "streaming":
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for tokens in tqdm(wiki.get_texts(), desc="Processing Wikipedia articles (streaming)"):
                    f.write(" ".join(tokens) + "\n")
        case "serialized":
            sentences = list(tqdm(wiki.get_texts(), desc="Processing Wikipedia articles (serialized)"))
            with open(checkpoint_path, "wb") as f:
                pickle.dump(sentences, f)
        case _:
            raise ValueError(f"Invalid processing mode: {mode}")
