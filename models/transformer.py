import os
import torch
import torch.nn as nn
from models.embedding import EmbeddingModel

class TransformerModel(EmbeddingModel):
    def __init__(self, fpath: str | os.PathLike, chars:list[str], context:int, feature_no:int=3, training_window:int=10000):
        """Initialisation function for model using embedding.
        
        Parameters
        ----------
            fpath : str | os.PathLike
                The path to the file containing training data for the model.
            chars : list[str]
                List of characters which appear in the training data.
            context : int   
                Size of the context window for predictions, as an integer.
            feature_no : int
                Dimensions of the embedding space where items of the vocabulary are encoded as vectors.
            training_window : int
                Number of lines from the training data on which the model is trained.
        """

        super().__init__(fpath=fpath, chars=chars, context=context, feature_no=feature_no, training_window=training_window)

    def _initialise_model(self, ):
        """Initialises the neural network architecture"""

        # Initialise model architecture
        self.embedding_matrix = nn.Embedding(num_embeddings=len(self.vocabulary), embedding_dim=self.feature_no)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.feature_no,
            nhead=4
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=3
        )

    def forward(self, X:torch.Tensor)->torch.Tensor:
        """Network forwards pass.
        
        Parameters 
        -------
            X : torch.Tensor

        Returns
        -------
            out : torch.Tensor
                Neural network outputs, in tensor form. These correspond to logits across all members of the vocabulary.
        """
        
        inputs = self.embedding_matrix(X)
        inputs = inputs.view(-1, self.context*self.feature_no)
        out = self.transformer(inputs)

        return out