import torch
from torch.utils.tensorboard import SummaryWriter
from utils import viz_weights_to_board, getActivation
import wandb
from datetime import datetime
import os

# internal training loop
def train(dataloader, model, loss_fn, optimizer, writer, dcon, mcon, tcon, epoch, noise_model, generator):
    size = len(dataloader.dataset)
    model.train()
    for batch, (corrupted, mask, uncorrupted, _) in enumerate(dataloader):
        corrupted = corrupted.to(tcon['device'])
        mask = mask.to(tcon['device'])
        uncorrupted = uncorrupted.to(tcon['device'])

        # compute prediction error
        reconstructed = model(corrupted)
        # compute loss: either use all elements or only the ones that were corrupted/masked
        if mcon['loss'] == 'full':
            loss = torch.sum(loss_fn(reconstructed, uncorrupted)) / torch.numel(reconstructed)
        elif mcon['loss'] == 'focused':
            loss = torch.sum(loss_fn(reconstructed, uncorrupted) * mask) / torch.sum(mask)
        else:
            raise ValueError('Loss type not recognized')
        
        if writer is not None and tcon['log']:
            writer.add_scalar("Loss_Batch/Train", loss, epoch*len(dataloader) + batch)
            wandb.log({"train_loss": loss})
        
        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(uncorrupted)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# internal validation loop
def validate(dataloader, model, loss_fn, writer, dcon, mcon, tcon, epoch, noise_model, generator):
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
    loss = 0

    with torch.no_grad():
        for i, (corrupted, mask, uncorrupted, _) in enumerate(dataloader):

            # set up hooks for saving activations
            hooks = []
            if i == 0 and image and tcon['activations'] and log:
                layer_iterator = enumerate(model.encoder.layers)
                for (k, layer) in layer_iterator:
                    if k % 2 != 0:
                        hooks.append(layer.register_forward_hook(getActivation(f'val_activations_layer_{k}', writer, img_index, epoch)))

            # set up actual data and targets and do forward pass
            corrupted = corrupted.to(tcon['device'])
            mask = mask.to(tcon['device'])
            uncorrupted = uncorrupted.to(tcon['device'])

            reconstructed = model(corrupted)

            # compute loss: either use all elements or only the ones that were corrupted/masked
            if mcon['loss'] == 'full':
                loss += torch.sum(loss_fn(reconstructed, uncorrupted)) / torch.numel(reconstructed)
            elif mcon['loss'] == 'focused':
                loss += torch.sum(loss_fn(reconstructed, uncorrupted) * mask) / torch.sum(mask)
            else:
                raise ValueError('Loss type not recognized')

            # write an example output, some weights and activations to tensorboard (can also write input if uncommented line)
            if i == 0 and image and log:
                if epoch == 1 or epoch == -1:
                    writer.add_image('val_img/target', torch.unflatten(uncorrupted.to('cpu'), 1, sizes)[img_index], epoch)
                    writer.add_image('val_img/input', torch.unflatten(corrupted.to('cpu'), 1, sizes)[img_index], epoch)
                    writer.add_image('val_img/mask', torch.unflatten(mask.to('cpu'), 1, sizes)[img_index], epoch)
                writer.add_image('val_img/prediction', torch.unflatten(reconstructed.to('cpu'), 1, sizes)[img_index], epoch)
                viz_weights_to_board(model, writer, epoch=epoch, subplot_number=1)
                for h in hooks:
                    h.remove()

    loss /= num_batches
    if log:
        wandb.log({"val_loss": loss})
        writer.add_scalar("Loss_Epoch/Validate", loss, epoch)
    print(f"Validation Avg loss: {loss} \n")


def train_imputation_model(model, loss_fn, optimizer, scheduler, 
                           dcon, mcon, tcon,
                           train_dataloader, validation_dataloader, test_dataloader=None,
                           noise_model=None):
    """
    Full training method for a (benchmark) denoising autoencoder model which is trained with the true missingness mechanism as corruption process.

    Parameters:
    model: nn.Module
        The model to be trained.
    loss_fn: nn.Module
        The loss function to be used for training.
    optimizer: torch.optim
        The optimizer to be used for training.
    scheduler: torch.optim.lr_scheduler
        The learning rate scheduler to be used for training.
    dcon: dict
        Dictionary containing the data configuration.
    mcon: dict
        Dictionary containing the model configuration.
    tcon: dict
        Dictionary containing the training configuration.
    train_dataloader: torch.utils.data.DataLoader
        DataLoader for the training set.
    validation_dataloader: torch.utils.data.DataLoader
        DataLoader for the validation set.
    test_dataloader: torch.utils.data.DataLoader
        DataLoader for the test set.
    noise_model: callable
        The noise model used for generating the missingness mechanism. Here: the true missingness mechanism function.
    """
    if not tcon['new_training']:
        return
    time = datetime.now().strftime("%B%d_%H_%M")
    writer = None

    generator = torch.Generator(device=torch.device(tcon['device'])).manual_seed(41)

    # make directory if it does not exist
    if tcon['save_model']:
        os.makedirs(f'models/{mcon["architecture"]}/{dcon["dataset"]}/{dcon["noise_mechanism"]}', exist_ok=True)
    
    if tcon['log']:
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
        train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, writer=writer, dcon=dcon, mcon=mcon, tcon=tcon, epoch=t, noise_model=noise_model, generator=generator)
        validate(dataloader=validation_dataloader, model=model, loss_fn=loss_fn, writer=writer, dcon=dcon, mcon=mcon, tcon=tcon, epoch=t, noise_model=noise_model, generator=generator)
        test(dataloader=test_dataloader, model=model, loss_fn=loss_fn, dcon=dcon, mcon=mcon, tcon=tcon, epoch=t, writer=writer) if test_dataloader is not None else None
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

# internal test loop
# test is for evaluating the performance of the imputation model on the actual missing values observations
# validation is for evaluating the reconstruction performance of the imputation model on the validation set created from the nona_dataset with corruption from MP model
def test(dataloader, model, loss_fn, dcon, mcon, tcon, epoch=-1, writer=None):
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
    loss = 0
    rmses = torch.zeros(len(dataloader.dataset))

    with torch.no_grad():
        for i, (NA_obs, mask, ground_truth, class_label) in enumerate(dataloader):
            # set up hooks for saving activations
            hooks = []
            if i == 0 and image and tcon['activations'] and log:
                layer_iterator = enumerate(model.encoder.layers)
                for (k, layer) in layer_iterator:
                    if k % 2 != 0:
                        hooks.append(layer.register_forward_hook(getActivation(f'test_activations_layer_{k}', writer, img_index, epoch)))

            # set up actual data and targets and do forward pass
            NA_obs = NA_obs.to(tcon['device'])
            mask = mask.to(tcon['device'])
            ground_truth = ground_truth.to(tcon['device'])
            mask = torch.flatten(mask, start_dim=1, end_dim=-1)

            reconstructed = model(NA_obs)

            imputed = NA_obs.clone()
            imputed[mask == 1] = reconstructed[mask == 1]

            # compute batch loss (only on missing values, i.e. focused)
            loss_batch = loss_fn(reconstructed, ground_truth) * mask
            individual_loss = torch.sum(loss_batch, dim=1) / torch.sum(mask, dim=1)
            rmses[(i*NA_obs.size(0)):((i+1)*NA_obs.size(0))] = torch.sqrt(individual_loss)
            loss += torch.sum(loss_batch) / torch.sum(mask)

            # write an example output, some weights and activations to tensorboard
            if i == 0 and image and log:
                if epoch == 1 or epoch == -1:
                    writer.add_image('test_img/ground_truth', torch.unflatten(ground_truth.to('cpu'), 1, sizes)[img_index], epoch)
                    writer.add_image('test_img/NA_obs', torch.unflatten(NA_obs.to('cpu'), 1, sizes)[img_index], epoch)
                writer.add_image('test_img/reconstructed', torch.unflatten(reconstructed.to('cpu'), 1, sizes)[img_index], epoch)
                writer.add_image('test_img/imputed', torch.unflatten(imputed.to('cpu'), 1, sizes)[img_index], epoch)
                viz_weights_to_board(model, writer, epoch=epoch, subplot_number=1)
                for h in hooks:
                    h.remove()

    loss /= num_batches
    if log:
        wandb.log({"test_loss": loss})
        writer.add_scalar("Loss_Epoch/Test", loss, epoch)
        writer.add_scalar("Loss_Epoch/Test_RMSE", torch.sqrt(loss), epoch)
        writer.add_scalar("Loss_Epoch/Test_STD_RMSE", torch.std(rmses), epoch)
    print(f"Test Avg loss: {loss}, RMSE: {torch.sqrt(loss)}, STD: {torch.std(rmses[~torch.isnan(rmses)])} \n")
    return loss
    

if __name__ == '__main__':
    None


    