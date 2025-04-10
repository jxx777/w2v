from pathlib import Path


class StreamingCorpus:
    """Streams tokenized sentences from a text file, processing one line at a time."""
    def __init__(self, txt_path: Path):
        self.txt_path = txt_path
        self._length = None

    def __iter__(self):
        with open(self.txt_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip().split()

    def __len__(self):
        if self._length is None:
            with open(self.txt_path, "r", encoding="utf-8") as f:
                self._length = sum(1 for _ in f)
        return self._length