import torch
from torch import nn

class SyntheticDenoisingAutoEncoder(nn.Module):
    """
    Synthetic Denoising Autoencoder model (in the thesis we call it the Adapted DAE (ADAE)). Is an imputation model trained on a learned missingness model.
    The same class is used for the Encoder model and the benchmark DAE model. The difference is the corruption process.

    Parameters
    ----------
    noise_model : torch.nn.Module
        Noise model. This parameter is not used within the class, but is included for logging purposes.
    layer_dims_enc : list
        List of encoder layer dimensions
    layer_dims_dec : list
        List of decoder layer dimensions
    relu : bool
        Use ReLU activation function. Default is True. If False, Sigmoid activation function is used.
    image : bool
        Input is an image. Default is True.
    """
    def __init__(self, noise_model, layer_dims_enc, layer_dims_dec, relu=True, image=True):
        super().__init__()
        self.layer_dims_enc = layer_dims_enc
        self.layer_dims_dec = layer_dims_dec
        self.image = image

        self.encoder = Encoder(layer_dims_enc, relu, image)
        self.decoder = Decoder(layer_dims_dec, relu)

    def forward(self, x):
        x_latent = self.encoder(x)
        x_reconstructed = self.decoder(x_latent)
        return x_reconstructed
        
    
class Encoder(nn.Module):
    """
    Part of the Synthetic Denoising Autoencoder model, the Encoder model.

    Parameters
    ----------
    layer_dims : list
        List of layer dimensions
    relu : bool
        Use ReLU activation function. Default is True. If False, Sigmoid activation function is used.
    image : bool
        Input is an image. Default is True.
    """
    def __init__(self, layer_dims, relu=True, image=True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_dims = layer_dims
        self.image = image

        #define layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(layer_dims[0], layer_dims[1])
        )
        self.layers.append(torch.nn.ReLU()) if relu else self.layers.append(torch.nn.Sigmoid())
        for i in range(1, len(layer_dims)-1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            self.layers.append(nn.ReLU()) if relu else self.layers.append(nn.Sigmoid())

    def forward(self, x):
        x = self.flatten(x) if self.image else x
        x_latent = self.layers(x)
        return x_latent
    
    
    
class Decoder(nn.Module):
    """
    Part of the Synthetic Denoising Autoencoder model, the Decoder model.

    Parameters
    ----------
    layer_dims : list
        List of layer dimensions
    relu : bool
        Use ReLU activation function. Default is True. If False, Sigmoid activation function is used.
    """
    def __init__(self, layer_dims, relu=True):
        super().__init__()
        self.layer_dims = layer_dims

        #define layers
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(layer_dims[0], layer_dims[1])
        )
        for i in range(1, len(layer_dims)-1):
            self.decoder.append(nn.ReLU()) if relu else self.decoder.append(nn.Sigmoid())
            self.decoder.append(nn.Linear(layer_dims[i], layer_dims[i + 1])
        )

    def forward(self, x):
        x_reconstructed = self.decoder(x)
        return x_reconstructed
    
# Based on the VAE implementation by Sebastian Raschka at github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-var.ipynb
class ImputeVAE(nn.Module):
    """
    Imputation Variational Autoencoder model.

    Parameters
    ----------
    layer_dims_enc : list
        List of encoder layer dimensions
    layer_dims_dec : list
        List of decoder layer dimensions
    dropout : float
        Dropout rate
    relu : bool
        Use ReLU activation function. Default is True. If False, Sigmoid activation function is used.
    image : bool
        Input is an image. Default is True.
    device : str
        Device to run the model on. Default is 'cuda'.
    """
    def __init__(self, layer_dims_enc, layer_dims_dec, dropout=0, relu=True, image=True, device='cuda'):
        super().__init__()
        self.device = device

        self.encoder = nn.Sequential(nn.Linear(layer_dims_enc[0], layer_dims_enc[1]))
        for i in range(1, len(layer_dims_enc)-2):
            self.encoder.append(nn.LeakyReLU(negative_slope=0.0001)) if relu else self.encoder.append(nn.Sigmoid())
            self.encoder.append(nn.Dropout(p=dropout))
            self.encoder.append(nn.Linear(layer_dims_enc[i], layer_dims_enc[i + 1]))

        self.encoder.append(nn.LeakyReLU(negative_slope=0.0001)) if relu else self.encoder.append(nn.Sigmoid())    

        self.z_mean = nn.Linear(layer_dims_enc[-2], layer_dims_enc[-1])
        self.z_log_var = nn.Linear(layer_dims_enc[-2], layer_dims_enc[-1])
        
        self.decoder = nn.Sequential(nn.Linear(layer_dims_dec[0], layer_dims_dec[1]))
        for i in range(1, len(layer_dims_dec)-1):
            self.decoder.append(nn.LeakyReLU(negative_slope=0.0001)) if relu else self.decoder.append(nn.Sigmoid())
            self.decoder.append(nn.Dropout(p=dropout))
            self.decoder.append(nn.Linear(layer_dims_dec[i], layer_dims_dec[i + 1]))
            
        self.decoder.append(nn.Sigmoid())
    
    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z

    def forward(self, features):
        encoder_hidden_rep = self.encoder(features)

        z_mean = self.z_mean(encoder_hidden_rep)
        z_log_var = self.z_log_var(encoder_hidden_rep)
        latent_rep = self.reparameterize(z_mean, z_log_var)

        decoded = self.decoder(latent_rep)
        return decoded