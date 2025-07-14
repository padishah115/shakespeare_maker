import os

# Training loops
from training.embedding import train_embedding
from training.ngram import train_ngram

def main(path: str | os.PathLike = "./training-data/shakespeare.txt"):
    
    path = path
    
    ############################
    # N-Grams of various sizes #
    ############################

    ngram_sizes = [3, 4, 5]
    for n in ngram_sizes:
        train_ngram(
            fpath=path,
            context=n
        )

    ###########################
    # Vector embedding models #
    ###########################

    features = [5, 10, 15] # dimensions of embedding space
    context = [5, 10, 15] # size of context window

    for f in features:
        for c in context:
            train_embedding(
                fpath=path,
                features=f,
                context=c
            )

        
        
    


if __name__ == "__main__":
    main()