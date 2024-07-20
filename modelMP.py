import torch
from torch import nn

class MaskPredMLP(nn.Module):
    """
    Mask Prediction fully-connected neural network model.

    Parameters
    ----------
    layer_dims : list
        List of layer dimensions
    dropout : float
        Dropout rate
    relu : bool
        Use ReLU activation function. Default is True. If False, Sigmoid activation function is used.
    image : bool
        Input is an image. Default is True.
    """
    def __init__(self, layer_dims, dropout=0, relu=True, image=True):
        super().__init__()
        self.layer_dims = layer_dims
        self.image = image

        self.layers = torch.nn.Sequential(torch.nn.Linear(layer_dims[0], layer_dims[1]))
        
        for i in range(1, len(layer_dims)-1):
            self.layers.append(torch.nn.ReLU()) if relu else self.layers.append(torch.nn.Sigmoid())
            self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

        self.layers.append(torch.nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)

# Based on the VAE implementation by Sebastian Raschka at github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-var.ipynb
class MaskPredVAE(nn.Module):
    """
    Mask Prediction Variational Autoencoder model.

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
    
# Based on the GAN implementation by Sebastian Raschka at github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan    
class MaskPredGAN(nn.Module):
    """
    Mask Prediction Generative Adversarial Network model.
    
    Parameters
    ----------
    layer_dims_gen : list
        List of generator layer dimensions
    layer_dims_disc : list
        List of discriminator layer dimensions
    dropout : float
        Dropout rate
    relu : bool
        Use ReLU activation function. Default is True. If False, Sigmoid activation function is used.
    image : bool
        Input is an image. Default is True.
    """
    def __init__(self, layer_dims_gen, layer_dims_disc, dropout=0, relu=True, image=True):
        super().__init__()
        self.generator = nn.Sequential(nn.Linear(layer_dims_gen[0], layer_dims_gen[1]))
        for i in range(1, len(layer_dims_gen)-1):
            #self.generator.append(nn.BatchNorm1d(layer_dims_gen[i]))
            self.generator.append(nn.LeakyReLU(negative_slope=0.0001)) if relu else self.generator.append(nn.Sigmoid())
            self.generator.append(nn.Dropout(p=dropout))
            self.generator.append(nn.Linear(layer_dims_gen[i], layer_dims_gen[i + 1]))
        #self.generator.append(nn.Tanh())
        self.generator.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(nn.Linear(layer_dims_disc[0], layer_dims_disc[1]))
        for i in range(1, len(layer_dims_disc)-1):
            self.discriminator.append(nn.BatchNorm1d(layer_dims_disc[i]))
            self.discriminator.append(nn.LeakyReLU(negative_slope=0.0001)) if relu else self.discriminator.append(nn.Sigmoid())
            self.discriminator.append(nn.Dropout(p=dropout))
            self.discriminator.append(nn.Linear(layer_dims_disc[i], layer_dims_disc[i + 1]))
        #self.discriminator.append(nn.Sigmoid())

            
    def generator_forward(self, z):
        img = self.generator(z)
        return img
    
    def discriminator_forward(self, img):
        pred = self.discriminator(img)
        return pred