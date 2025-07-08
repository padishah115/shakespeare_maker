import os

# Import language models
from models.ngram import NGram

def main(path: str | os.PathLike = "shakespeare.txt"):
    
    path = path

    # Bigram model
    print("\n")
    print("BIGRAM MODEL")
    print("------")
    bigram = NGram(fname=path, context=2)
    bigram.generate_text(lines_to_gen=10)
    print("------")
    
    # Trigram model
    print("\n")
    print("TRIGRAM MODEL")
    print("------")
    trigram = NGram(fname=path, context=3)
    trigram.generate_text(lines_to_gen=10)
    print("------")

    # 4gram model
    print("\n")
    print("4-GRAM MODEL")
    print("------")
    trigram = NGram(fname=path, context=4)
    trigram.generate_text(lines_to_gen=10)
    print("------")


if __name__ == "__main__":
    main()