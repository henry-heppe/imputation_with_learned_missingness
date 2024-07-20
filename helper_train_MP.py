import torch
from torch.utils.tensorboard import SummaryWriter
from utils import viz_weights_to_board, getActivation
import wandb
from datetime import datetime
import os

# internal training loop
def train(dataloader, model, encoder, loss_fn, optimizer, writer, dcon, mcon, tcon, epoch):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y, z, w) in enumerate(dataloader):
        X = X.to(tcon['device'])
        y = y.to(tcon['device'])
        y = torch.flatten(y, start_dim=1, end_dim=-1)

        representation = encoder(X) if encoder is not None else X
        # compute prediction error
        pred = model(representation)
        loss = loss_fn(pred, y)
        batch_accuracy = ((pred > 0.5) == y.bool()).type(torch.float).sum().item() / torch.numel(y)
        if writer is not None:
            writer.add_scalar("Loss_Batch/Train", loss, epoch*len(dataloader) + batch)
            writer.add_scalar("Accuracy_Batch/Train", batch_accuracy, epoch*len(dataloader) + batch)
        
        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} accuracy: {batch_accuracy:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, encoder, loss_fn, dcon, mcon, tcon, writer=None, epoch=-1):
    """
    Testing loop for the Mask Predictor (MP) model. This function is called after each epoch of training.

    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
        Dataloader for testing data with missingness.
    encoder: nn.Module
        The Encoder to be used in front of the model.
    loss_fn: nn.Module
        The loss function to be used for testing.
    model: nn.Module
        The model to be tested.
    dcon: dict
        Dictionary containing the data configuration.
    mcon: dict
        Dictionary containing the model configuration.
    tcon: dict
        Dictionary containing the training configuration.
    writer: torch.utils.tensorboard.SummaryWriter
        The tensorboard writer. Default is None.
    epoch: int
        The current epoch number. Default is -1 (no epoch).

    Returns
    -------
    None
    """
    size = len(dataloader.dataset)
    numel = torch.numel(dataloader.dataset.y)
    num_batches = len(dataloader)
    img_index = tcon['img_index']
    image = mcon['image']
    log = tcon['log']
    if writer is None:
        log = False

    if 'MNIST' in dcon['dataset']:
        sizes = [1, 28, 28]
    elif 'CIFAR' in dcon['dataset']:
        sizes = [1, 32, 32]

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for i, (X, y, z, w) in enumerate(dataloader):

            # set up hooks for saving activations
            hooks = []
            if i == 0 and image and tcon['activations'] and log:
                layer_iterator = enumerate(model.layers) if hasattr(model, 'layers') else enumerate(model.encoder)
                for (k, layer) in layer_iterator:
                    if k % 2 != 0:
                        hooks.append(layer.register_forward_hook(getActivation(f'activations_layer_{k}', writer, img_index, epoch)))

            # set up actual data and targets and do forward pass
            X = X.to(tcon['device'])
            y = y.to(tcon['device'])
            y = torch.flatten(y, start_dim=1, end_dim=-1)

            representation = encoder(X) if encoder is not None else X
            pred = model(representation)
            
            test_loss += loss_fn(pred, y).item()
            correct += ((pred > 0.5) == y.bool()).type(torch.float).sum().item()

            # write an example output, some weights and activations to tensorboard
            if i == 0 and image and log:
                if epoch == 1 or epoch == -1:
                    writer.add_image('target', torch.unflatten(y.to('cpu'), 1, sizes)[img_index], epoch)
                    writer.add_image('input', torch.unflatten(X.to('cpu'), 1, sizes)[img_index], epoch)
                writer.add_image('prediction', torch.unflatten(pred.to('cpu'), 1, sizes)[img_index], epoch)
                viz_weights_to_board(model, writer, epoch=epoch, subplot_number=1)
                for h in hooks:
                    h.remove()

    test_loss /= num_batches
    test_accuracy = correct / numel
    if log:
        wandb.log({"loss": test_loss})
        wandb.log({"accuracy": test_accuracy})
        writer.add_scalar("Loss_Epoch/Validate", test_loss, epoch)
        writer.add_scalar("Accuracy_Epoch/Validate", test_accuracy, epoch)
    print(f"Validation Avg loss: {test_loss} Validation Accuracy: {test_accuracy} \n")


def train_model(model, encoder, loss_fn, optimizer, scheduler, 
                           dcon, mcon, tcon,
                           train_dataloader, validation_dataloader, test_dataloader=None):
    """
    Full training method for fully-connected neural network as a Mask Predictor (MP) model.

    Parameters:
    model: nn.Module
        The model to be trained.
    encoder: nn.Module
        The Encoder to be used in front of the model.
    loss_fn: nn.Module
        The loss function to be used for training.
    optimizer: torch.optim
        The optimizer to be used for training.
    scheduler: torch.optim.lr_scheduler
        The scheduler to be used for training.
    dcon: dict
        Dictionary with data configuration.
    mcon: dict
        Dictionary with model configuration.
    tcon: dict
        Dictionary with training configuration.
    train_dataloader: torch.utils.data.DataLoader
        DataLoader for training data.
    validation_dataloader: torch.utils.data.DataLoader
        DataLoader for validation data.
    test_dataloader: torch.utils.data.DataLoader
        DataLoader for test data.
    """
    if not tcon['new_training']:
        return
    time = datetime.now().strftime("%B%d_%H_%M")
    writer = None
    # make directory if it does not exist
    if tcon['save_model']:
        os.makedirs(f'models/{mcon["architecture"]}/{dcon["dataset"]}/{dcon["noise_mechanism"]}', exist_ok=True)
    
    if tcon['log']:
            #wandb.tensorboard.patch(root_logdir=f'runs/')
            wandb.init(
                # set the wandb project where this run will be logged
                project="denoising_autencoders_logs",
                sync_tensorboard=True,
                save_code=True,
                tags=[mcon['architecture'], dcon['dataset'], dcon['noise_mechanism']],
                name=f'{time}',
                # track hyperparameters and run metadata
                config={
                    'data_config': dcon,
                    'model_config': mcon,
                    'train_config': tcon,
                }
            )
            writer = SummaryWriter(log_dir=f'runs/{mcon['architecture']}/{dcon['dataset']}/{dcon["noise_mechanism"]}/{time}')

    # training/validation loop
    for t in range(mcon['epochs']):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dataloader=train_dataloader, model=model, encoder=encoder, loss_fn=loss_fn, optimizer=optimizer, writer=writer, dcon=dcon, mcon=mcon, tcon=tcon, epoch=t)
        test(dataloader=validation_dataloader, model=model, encoder=encoder, loss_fn=loss_fn, dcon=dcon, mcon=mcon, tcon=tcon, writer=writer, epoch=t)
        scheduler.step() if mcon['lr_decay'] else None
        if writer is not None:
            writer.add_scalar("LR", scheduler.get_last_lr()[0], t)
    print("Done!")

    if tcon['save_model']:
        torch.save({'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'tcon': tcon,
                    'mcon': mcon,
                    'dcon': dcon},
                    f'models/{mcon['architecture']}/{dcon['dataset']}/{dcon["noise_mechanism"]}/model{time}.pth')
    #wandb.save(f'models/{mcon['architecture']}/{dcon['dataset']}/model{time}.pth')
    
    if tcon['log']:
        viz_weights_to_board(model, writer, subplot_number=8) if mcon['image'] else None
        writer.flush()
        writer.close()
        wandb.finish()    

if __name__ == '__main__':
    None


    