import os

# Import language models
from models.ngram import NGram

def main(path: str | os.PathLike = "shakespeare.txt"):
    
    path = path

    # Bigram model
    bigram = NGram(fname=path, context=2)
    bigram.generate_text(lines_to_gen=10)
    
    # Trigram model
    # trigram = NGram(fname=path, context=3)
    # trigram.generate_text(lines_to_gen=10)


if __name__ == "__main__":
    main()