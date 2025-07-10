import os

# Import language models
from models.ngram import NGram
from models.embedding import EmbeddingModel

def main(path: str | os.PathLike = "shakespeare.txt"):
    
    path = path

    # # 4- and 5-gram model
    # for i in range(4, 6):
    #     print("\n")
    #     print(f"{i}-GRAM MODEL")
    #     print("------")
    #     bigram = NGram(fpath=path, context=i)
    #     bigram.generate_text(lines_to_gen=10)
    #     print("\n")
    #     print("------")

    # Vector embedding models
    for i in range(9, 11):
        print("\n")
        print(f"EMBEDDING MODEL, {i} FEATURES")
        print("------")
        embedding_model = EmbeddingModel(fpath=path, context=3, feature_no=i)
        embedding_model.generate_text(lines_to_gen=10, n_epochs=1000)
        embedding_model.save_model(f"./trained-models-statedicts/{i}-feature_embedding_model.pt")
        print("\n")
        print("------")
        
    


if __name__ == "__main__":
    main()