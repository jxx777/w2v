import pickle
from pathlib import Path


class SerializedCorpus:
    """Loads the entire corpus from a serialized pickle file into memory."""
    def __init__(self, pickle_path: Path):
        self.pickle_path = pickle_path
        self._sentences = None

    def _load(self):
        if self._sentences is None:
            with open(self.pickle_path, "rb") as f:
                self._sentences = pickle.load(f)
        return self._sentences

    def __iter__(self):
        for sentence in self._load():
            yield sentence

    def __len__(self):
        return len(self._load())