import os

# Training loops
from training.embedding import train_embedding
from training.ngram import train_ngram

def get_chars(lines:list[str], training_window)->list[str]:
        """Produces list of characters from a given list of lines.
        
        Parameters
        ----------
            lines : list[str]
                List of lines in the text document as strings.

        Returns
        -------
            chars : list[str]
        """

        chars = []

        # Train on the specified number of lines in the file
        for line in lines[:training_window]:
            splitline = list(line.split(" ")) # introduce special '£' character for linebreak
            words = ['+' + word for word in splitline if word != ''] + ["£"] # introduce '+' character for space
            for word in words:
                chars += ([char for char in word])

        return chars

def main(path: str | os.PathLike = "./training-data/shakespeare.txt"):
    
    training_window = 10000
    path = path

    lines = open(path).read().splitlines()
    chars = get_chars(lines=lines, training_window=training_window)
    
    ############################
    # N-Grams of various sizes #
    ############################

    # ngram_sizes = [3, 4, 5]
    # for n in ngram_sizes:
    #     train_ngram(
    #         fpath=path,
    #         context=n
    #     )

    ###########################
    # Vector embedding models #
    ###########################

    # features = [5, 10, 15] # dimensions of embedding space
    # context = [5, 10, 15] # size of context window

    # for f in features:
    #     for c in context:
    #         train_embedding(
    #             fpath=path,
    #             model_type="EMBEDDING",
    #             chars=chars,
    #             features=f,
    #             context=c
    #         )

    features = [64, 128, 256] # dimensions of embedding space
    context = [5, 10, 15] # size of context window

    for f in features:
        for c in context:
            train_embedding(
                fpath=path,
                model_type="TRANSFORMER",
                chars=chars,
                features=f,
                context=c
            )
        
        
    


if __name__ == "__main__":
    main()