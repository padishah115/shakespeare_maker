import os

# Import language models
from models.ngram import NGram

def main(path: str | os.PathLike = "shakespeare.txt"):
    
    path = path

    # Bigram model
    for i in range(4, 6):
        print("\n")
        print(f"{i}-GRAM MODEL")
        print("------")
        bigram = NGram(fname=path, context=i)
        bigram.generate_text(lines_to_gen=10)
        print("\n")
        print("------")
    


if __name__ == "__main__":
    main()