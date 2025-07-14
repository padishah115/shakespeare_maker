import os

# Import language models
from models.ngram import NGram
from models.embedding import EmbeddingModel

def main(path: str | os.PathLike = "./training-data/shakespeare.txt"):
    
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

    features = 5

    for context in [5, 10, 15]:
        print("\n")
        print(f"EMBEDDING MODEL, {features} FEATURES, CONTEXT WINDOW: {context}")
        print("------")
        embedding_model = EmbeddingModel(fpath=path, context=context, feature_no=features)
        embedding_model.generate_text(lines_to_gen=10, n_epochs=2000)
        embedding_model.save_text(f"./generated-text/{features}-feature-{context}-context.txt")
        embedding_model.save_model(f"./trained-models-statedicts/{features}-feature_embedding_model_context_{context}.pt")
        print("\n")
        print("------")
        
    


if __name__ == "__main__":
    main()