from nltk.tokenize import sent_tokenize, word_tokenize
from unidecode import unidecode

def clean_article(text):
    sentences = []
    for sent in sent_tokenize(" ".join(text)):
        tokens = word_tokenize(sent.lower())
        tokens = [unidecode(t) for t in tokens if t.isalpha() and len(t) > 3]
        if tokens:
            sentences.append(tokens)
    return sentences
