import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import time


class EmbeddingModel(nn.Module):
    """More advanced model which uses embedding of letters in order to generate more sophisticated predictions."""

    def __init__(self, fpath: str | os.PathLike, context:int, feature_no:int=3, training_window:int=10000):
        """Initialisation function for model using embedding.
        
        Parameters
        ----------
            fpath : str | os.PathLike
                The path to the file containing training data for the model.
            context : int   
                Size of the context window for predictions, as an integer.
            feature_no : int
                Dimensions of the embedding space where items of the vocabulary are encoded as vectors.
            training_window : int
                Number of lines from the training data on which the model is trained.
        """

        super().__init__()

        self.fpath = fpath
        self.context = context
        self.feature_no = feature_no
        self.training_window = training_window


    def _get_chars(self, lines:list[str])->list[str]:
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
        for line in lines[:self.training_window]:
            splitline = list(line.split(" ")) # introduce special '£' character for linebreak
            words = ['+' + word for word in splitline if word != ''] + ["£"] # introduce '+' character for space
            for word in words:
                chars += ([char for char in word])

        return chars
    

    def _set_mapping(self, ):
        """Generates a vocabulary for the class composed of all characters in the text, and generates both string-to-index and index-to-string
        mapping dictionaries, where the index is unique for each character in the vocabulary."""
        
        lines = open(self.fpath).read().splitlines()

        # Get character list
        self.chars = self._get_chars(lines=lines)

        # Generate mapping dictionaries
        self.vocabulary = sorted(set(self.chars))
        self.stoi = {s:i for i, s in enumerate(self.vocabulary)}
        self.itos = {i:s for s, i in self.stoi.items()}


    def _generate_training_data(self, )->tuple[torch.Tensor, torch.Tensor]:
        """Generates training data in the form of two matrices: a matrix of indices, X, and a matrix of labels, Y.
        
        Returns
        -------
            X : torch.Tensor
                Tensor encoding the sequence of characters (as indices) in the training data
            Y : torch.Tensor
                Tensor encoding labels, i.e. the index of the character that follows the character in the corresponding position in X.
        """
        
        X, Y = [], [] #X are contexts, Y are what we should predict
        context = [0] * self.context # intial window is empty
        for ch in self.chars:
            # Get index of character
            ix = self.stoi[ch]

            #Store context windows
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] #modify context window by shifting one word over

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        return X, Y


    def _initialise_model(self, ):
        """Initialises the neural network architecture"""

        # Initialise model architecture
        self.embedding_matrix = nn.Embedding(num_embeddings=len(self.vocabulary), embedding_dim=self.feature_no)
        self.fc1 = nn.Linear(in_features=self.context*self.feature_no, out_features=400)
        self.fc2 = nn.Linear(in_features=400, out_features=len(self.vocabulary))
        self.activation = torch.tanh

        # Initialise random embedding matrix of size V x feature_no
        V = len(self.vocabulary) # size of vocabulary


    def _initialise_optimizer(self, ):
        """Initialises the Adam optimizer using the model's parameters."""

        self.optimizer = optim.Adam(self.parameters())


    def _forward_pass(self, X:torch.Tensor)->torch.Tensor:
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
        out = self.fc1(inputs)
        out = self.activation(out)
        out = self.fc2(out)

        return out

    
    def _backward_pass(self, outputs, targets)->float:
        """Network backwards pass. Computes loss and performs backpropagation.
        
        Parameters
        ----------
            outputs : torch.Tensor
                Model outputs in tensor form.
            targets : torch.Tensor
                Ground-truth class label.

        Returns
        -------
            loss : float
                Loss at the epoch.
        """ 

        # Calculate loss, clear gradients, differentiate and step.
        loss = F.cross_entropy(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def _train_model(self, X:torch.Tensor, Y:torch.Tensor, n_epochs:int=100):
        """Training cycle for the model.
        
        Parameters
        ----------
            X : torch.Tensor
            Y : torch.Tensor
                Training data ground-truth labels
            n_epochs : int
                Number of epochs over which we train the model.
        """
        epochs = []
        losses = []
        times = []
        for epoch in range(1, n_epochs+1):
            ti = time.time() #time length of each epoch
            outputs = self._forward_pass(X=X)
            targets = Y
            loss = self._backward_pass(outputs, targets)
            tf=time.time() # time length of each epoch
            dt = tf-ti
            
            if epoch % 500 == 0 or epoch == 1:
                print(f"Loss at epoch {epoch}: {loss:.4f}")
            
            # Append epoch number and loss for analysis
            epochs.append(epoch)
            losses.append(loss)
            times.append(dt)

        # Save loss statistics
        loss_data = {"Epoch":epochs, "Loss":losses, "Time":times}
        loss_df = pd.DataFrame(loss_data)
        loss_df.to_csv(f"./loss-stats/{self.feature_no}-feature_embedding_model_context_{self.context}.csv")

        print("<Concluded training>.")
        print("------")


    def _get_generated_lines(self, lines_to_gen:int, seed:int=42)->list[str]:
        """Returns a list of lines generated by the model.
        
        Parameters
        ----------
            lines_to_gen : int
                Number of lines that the model will generate.
            seed : int
                Seed for the generator used to sample from probability distributions.

        Returns
        -------
            generated_lines : list[str]
                All lines generated by the model as strings.
        """

        generated_lines = []
        generator = torch.Generator().manual_seed(seed)


        context = [self.stoi["£"]] * self.context
        for i in range(lines_to_gen):
            new_line = False
            line = []
            while not new_line:

                prob_row = F.softmax(self._forward_pass(torch.tensor(context)), dim=1).view(-1)

                ix = torch.multinomial(prob_row, generator=generator, num_samples=1)

                context = context[1:] + [ix.item()]

                # if the next character is a space, append a space
                if ix.item() == self.stoi["+"]:
                    next_char = ' '
                    line.append(next_char)
                
                # if the next character is the break character, add a new line
                elif ix.item() == self.stoi["£"]:
                    i = i + 1
                    generated_lines.append(line)
                    new_line = True

                # Else, just eppane dht character
                else:
                    next_char = self.itos[ix.item()]
                    line.append(next_char)


        return generated_lines


    def generate_text(self, lines_to_gen:int, n_epochs:int):
        """
        Parameters
        ----------
            lines_to_gen : int
                Number of lines of text which we want the model to generate
            n_epochs : int
                Number of epochs for which the training cycle will be executed.
        """

        # Generate string-to-integer and integer-to-string mappings
        self._set_mapping()

        # Set model and train
        self._initialise_model()
        self._initialise_optimizer()

        # Generate training data
        X, Y = self._generate_training_data()

        # Train model
        self._train_model(X=X, Y=Y, n_epochs=n_epochs)

        # Get list of bogus lines generated by model
        generated_lines = self._get_generated_lines(lines_to_gen=lines_to_gen, seed=42)
        for line in generated_lines:
            full_line = ''.join(letter for letter in line)
            print(full_line)

    def save_model(self, spath:str | os.PathLike):
        """Saves the model's state dictionary at a supplied path.
        
        Parameters
        ----------
            spath : str | os.PathLike
                The path at which the model's .pt state dict will be saved.
        """

        torch.save(self.state_dict(), f=spath)