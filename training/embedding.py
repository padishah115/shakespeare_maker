import os
from models.embedding import EmbeddingModel


def train_embedding(fpath:str | os.PathLike, 
                    features:int, 
                    context:int, 
                    epochs:int=2000, 
                    lines_to_gen:int=10,
                    spath_text:str | os.PathLike="./generated-text-embedding",
                    spath_model:str | os.PathLike="./trained-models-statedicts"
                    ):
    """Train an embedding model with a given number of features and a given context window size.
    Also specify the number of epochs over which the model is trained.
    
    Parameters
    ----------
        fpath : str | os.PathLike
            Path to the file containing training data.
        features : int
            Number of features we would like to include in the model, i.e. the dimensions of the embedding space.
        epochs : int
            The number of epochs over which training of the model is performed.
        lines_to_gen : int
            The number of lines of text we would like the model to generate.
        spath_text : str | os.PathLike
            The path at which text generated by the model will be saved.
        spath_model : str | os.PathLike
            The path at which the model's state dictionary will be saved.
    """

    print("\n")
    print(f"EMBEDDING MODEL, {features} FEATURES, CONTEXT WINDOW: {context}")
    print("------")
    embedding_model = EmbeddingModel(fpath=fpath, context=context, feature_no=features)
    embedding_model.generate_text(lines_to_gen=lines_to_gen, n_epochs=epochs)
    embedding_model.save_text(f"{spath_text}/{features}-feature-{context}-context.txt")
    embedding_model.save_model(f"{spath_model}/{features}-feature_embedding_model_context_{context}.pt")
    print("\n")
    print("------")