import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """
    A simple neural network template for self-supervised learning.

    Structure:
    1. Encoder: Maps an input image of shape 
       (input_channels, input_dim, input_dim) 
       into a lower-dimensional feature representation.
    2. Projector: Transforms the encoder output into the final 
       embedding space of size `proj_dim`.

    Notes:
    - DO NOT modify the fixed class variables: 
      `input_dim`, `input_channels`, and `feature_dim`.
    - You may freely modify the architecture of the encoder 
      and projector (layers, activations, normalization, etc.).
    - You may add additional helper functions or class variables if needed.
    """

    ####### DO NOT MODIFY THE CLASS VARIABLES #######
    input_dim: int = 64
    input_channels: int = 3
    feature_dim: int = 1000
    proj_dim: int = 128
    #################################################

    def __init__(self):
        super().__init__()


        ######################## TODO: YOUR CODE HERE ########################
        # Define the layers of the encoder and projector here

        # # Encoder: flattens the image and learns a compact feature representation
        # self.encoder = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(self.input_channels * self.input_dim * self.input_dim, self.feature_dim),
        #     nn.ReLU(),
        # )

        # using a resnet-50 backbone
        resnet = models.resnet50(weights=None)
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules) 

        self.flatten = nn.Flatten()
        self.neck = nn.Linear(2048, self.feature_dim, bias=True)

        # Projector: maps encoder features into the final embedding space
        # self.projector = nn.Linear(self.feature_dim, self.proj_dim)

        # implementing a 3-layer MLP
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.proj_dim, bias=True)
        )
        
        ######################################################################

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape 
                              (batch_size, input_channels, input_dim, input_dim).

        Returns:
            torch.Tensor: Output embedding of shape (batch_size, proj_dim).
        """
        h = self.encoder(x)            # (B, 2048, 1, 1)
        h = self.flatten(h)            # (B, 2048)
        features = self.neck(h)        # (B, 1000) == (B, feature_dim)
        
        projected_features = self.projector(features)  # (batch_size, proj_dim)
        projected_features = F.normalize(projected_features, dim=-1)
        return features, projected_features
    
    def get_features(self, x):
        """
        Get the features from the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape 
                              (batch_size, input_channels, input_dim, input_dim).

        Returns:
            torch.Tensor: Output features of shape (batch_size, feature_dim).
        """
        h = self.encoder(x)           
        h = self.flatten(h)            
        features = self.neck(h)
        return features