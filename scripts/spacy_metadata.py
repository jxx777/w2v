import os
import spacy
from tqdm import tqdm

# Load spaCy's model globally
nlp = spacy.load("ro_core_news_lg")

def extract_metadata(sentences, batch_size: int = 100):
    """
    Extracts metadata for each token in the given sentences in parallel using spaCy's nlp.pipe.

    Parameters:
        sentences (List[str] or List[List[str]]): List of sentences (as strings) or lists of tokens.
        batch_size (int): Number of texts to process per batch.

    Returns:
        dict: A dictionary mapping each normalized word to its metadata,
              including POS, detailed tag, dependency, lemma, morphological features, entity type, and frequency count.
    """
    # Convert each sentence to a string if it is a list of tokens.
    texts = [ " ".join(sentence) if isinstance(sentence, list) else sentence for sentence in sentences ]

    metadata_dict = {}
    n_process = os.cpu_count()  # Use all available CPUs

    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process),
                    total=len(texts),
                    desc="Extracting metadata with spaCy"):
        for token in doc:
            word = token.text.lower().strip()
            if not word or not token.is_alpha:  # Optionally ignore non-alphabetic tokens
                continue

            if word not in metadata_dict:
                metadata_dict[word] = {
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                    "lemma": token.lemma_,
                    "morph": str(token.morph),
                    "ent_type": token.ent_type_,
                    "frequency": 1
                }
            else:
                metadata_dict[word]["frequency"] += 1
    return metadata_dict
