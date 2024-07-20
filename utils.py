import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import os
import shutil
import data
import modelSR
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


def viz_weights_to_board(model, writer, epoch=-1, subplot_number=6):
    """
    Visualize weights of the first layer of the model to tensorboard. Each subplots represents the weights for one neuron.

    Parameters
    ----------
    model : torch.nn.Module
        Model to visualize
    writer : torch.utils.tensorboard.SummaryWriter
        Tensorboard writer
    epoch : int
        Epoch number
    subplot_number : int
        Number of subplots in the figure

    Returns
    -------
    None
    """
    k = 0
    for name, param in model.named_parameters():
        k += 1
        if len(param.size()) < 2 or 'noise_model' in name: # filter out bias and batchnorm layers and noise model
            k -= 1
            continue

        if param.size()[1] == 784:
            img_size = [28, 28]
        elif param.size()[1] == 1024:
            img_size = [32, 32]
        else:
            continue
    
        fig = plt.figure(num=1, figsize=(10,10))
        for i in range(subplot_number**2):
            neuron = int(i * param.size()[0] / (subplot_number**2))
            pixels = torch.unflatten(param[neuron], 0, img_size).to('cpu').detach().numpy()
            fig.add_subplot(subplot_number,subplot_number,i+1)
            plt.imshow(pixels, cmap='gray')
            plt.axis('off')

        writer.add_figure(f'WeightsLayer{k}', fig, epoch)     


def getActivation(name, writer, img_index, epoch = -1):
    """
    Create a hook to visualize activations of a layer in the model to tensorboard.

    Parameters
    ----------
    name : str
        Name of the layer
    writer : torch.utils.tensorboard.SummaryWriter
        Tensorboard writer
    img_index : int
        Index of the image to visualize
    epoch : int
        Epoch number

    Returns
    -------
    function
        Hook function
    """
    def hook(model, input, output):
        #input_transf = torch.unflatten(input[0], 1, [1, 28, 28]).to('cpu').numpy()[1][0]
        output_batch = output.to('cpu').numpy()
        writer.add_scalar(f'nonzero_{name}', np.count_nonzero(output_batch)/(output.size()[0]*output.size()[1]), epoch)

        if 784 in output.size():
            output_transf = torch.unflatten(output, 1, [1, 28, 28]).to('cpu').numpy()[img_index][0]
        elif 1024 in output.size():
            output_transf = torch.unflatten(output, 1, [1, 32, 32]).to('cpu').numpy()[img_index][0]
        else:
            return
        fig = plt.figure(num=1, figsize=(10,10))
        plt.imshow(output_transf, cmap='gray')
        writer.add_figure(name, fig, epoch)
    return hook

def remove_last_log(path):
    """
    Remove the last tensorboard log file in the directory.

    Parameters
    ----------
    path : str
        Path to the directory

    Returns
    -------
    None
    """
    shutil.rmtree(os.path.join(path, sorted(os.listdir(path), reverse=True)[0]))

def compute_NA_loss(reconstruction, ground_truth, mask):
    """
    Compute the focused loss for the imputation task.

    Parameters
    ----------
    reconstruction : torch.Tensor
        Reconstructed data
    ground_truth : torch.Tensor
        Ground truth data
    mask : torch.Tensor
        Mask of missing values

    Returns
    -------
    float
        Loss
    """
    return torch.sum(torch.square(reconstruction.cpu() - ground_truth.cpu()) * mask.cpu()) / torch.sum(mask.cpu())

# Adapted from github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/basic-ml/softmax-regression.ipynb
def softmaxRegression(X, y, num_classes, num_epochs=100, device='cuda'):
    """
    Downstream task for the imputation task. Train a classifier on the imputed data.

    Parameters
    ----------
    X : torch.Tensor
        Imputed data
    y : torch.Tensor
        Ground truth labels
    num_classes : int
        Number of classes
    num_epochs : int
        Number of epochs
    device : str
        Device to use

    Returns
    -------
    float
        Validation accuracy
    """
    X = X.to(device)
    y = y.to(device).unsqueeze(1)
    model = modelSR.SoftmaxRegression(num_features=X.size(1), num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    downstream_data_train, downstream_data_val, downstream_targets_train, downstream_targets_val = train_test_split(X, y, test_size=0.4, random_state=0)
    train_loader = DataLoader(data.DatasetZipped(downstream_data_train, downstream_targets_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(data.DatasetZipped(downstream_data_val, downstream_targets_val), batch_size=64, shuffle=False)


    for epoch in range(num_epochs):
        for batch_idx, (features, targets) in enumerate(train_loader):
            
            features = features.to(device)
            targets = targets.to(device)
                
            logits, probas = model(features)
            
            cost = F.cross_entropy(logits, targets.squeeze(1))
            optimizer.zero_grad()
            cost.backward()
            
            optimizer.step()
            
        with torch.set_grad_enabled(False):
            model.eval()
            correct, total = 0, 0
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                logits, probas = model(features)
                _, predicted_labels = torch.max(probas, 1)
                total += targets.size(0)
                correct += (predicted_labels == targets.squeeze(1)).sum()
            accuracy = correct / total
    return accuracy
    
if __name__ == '__main__':
    None