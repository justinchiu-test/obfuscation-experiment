import re
from collections import Counter
def clean_text(text):
    return re.findall(r"[a-z0-9]+", text.lower())
def word_frequencies(words):
    return Counter(words)
def top_k_words(freqs, k=10):
    return freqs.most_common(k)
def main(filename):
    with open(filename) as f:
        words = clean_text(f.read())
    freqs = word_frequencies(words)
    for word, count in top_k_words(freqs, 10):
        print(word, count)
if __name__ == "__main__":
    main("example.txt")
