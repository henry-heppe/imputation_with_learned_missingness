import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from utils import viz_weights_to_board, getActivation
import wandb
from datetime import datetime
import os
import matplotlib.pyplot as plt

### Based on the GAN implementation by Sebastian Raschka at github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan

# internal training loop
def train(na_dataloader, nona_dataloader, both_loader, model, optimizer_gen, optimizer_disc, writer, dcon, mcon, tcon, epoch, random_generator=None):
    size = len(na_dataloader.dataset)
    model.train()
    for batch, (nona_input, _, _, _, in_corrupted, in_mask, in_ground_truth, _) in enumerate(both_loader):
        nona_input = nona_input.to(tcon['device'])
        nona_input = torch.flatten(nona_input, start_dim=1, end_dim=-1)
        in_mask = in_mask.to(tcon['device']).bool()
        in_mask = torch.flatten(in_mask, start_dim=1, end_dim=-1)
        in_corrupted = in_corrupted.to(tcon['device'])
        in_corrupted = torch.flatten(in_corrupted, start_dim=1, end_dim=-1)

        valid = torch.ones(in_mask.size(0), 1, device=tcon['device']).float()
        fake = torch.zeros(in_mask.size(0), 1, device=tcon['device']).float()

        # Generate new masks
        noise = torch.randn((nona_input.size(0), mcon['layer_dims_gen'][0]-nona_input.size(1)), generator=random_generator, device=tcon['device'])
        conditional_noise = torch.cat((nona_input, noise), dim=1)
        gen_mask = model.generator_forward(conditional_noise)

        # ----- Train the discriminator -----
        optimizer_disc.zero_grad()

        # get discriminator loss on real images
        true_noisy = in_corrupted.clone()
        disc_pred_real = model.discriminator_forward(torch.cat((true_noisy, in_mask), dim=1))
        real_loss = F.binary_cross_entropy_with_logits(disc_pred_real, valid)

        # get discriminator loss on fake images
        gen_corrupted = (1-gen_mask) * nona_input.clone() + gen_mask * dcon['replacement']
        disc_pred_fake = model.discriminator_forward(torch.cat((gen_corrupted.detach(), gen_mask.detach()), dim=1))
        fake_loss = F.binary_cross_entropy_with_logits(disc_pred_fake, fake)

        # combined loss
        disc_loss = (real_loss + fake_loss)*0.5
        disc_loss.backward()

        optimizer_disc.step()

        # ----- Train the generator -----
        optimizer_gen.zero_grad()

        # get discriminator loss on fake images with flipped labels
        disc_pred_fake = model.discriminator_forward(torch.cat((gen_corrupted, gen_mask), dim=1))
        gen_loss = F.binary_cross_entropy_with_logits(disc_pred_fake, valid)
        gen_loss.backward()

        optimizer_gen.step()

        if writer is not None:
            writer.add_scalar("Generator_Loss_Batch/Train", gen_loss, epoch*len(both_loader) + batch)
            writer.add_scalar("Discriminator_Loss_Batch/Train", disc_loss, epoch*len(both_loader) + batch)
            if batch == 0 and epoch in [0, 9]:
                if 'MNIST' in dcon['dataset']:
                    sizes = [1, 28, 28]
                elif 'CIFAR' in dcon['dataset']:
                    sizes = [1, 32, 32]

                img_index = 3
                pixels = torch.squeeze(torch.unflatten(true_noisy, 1, sizes), 1).to('cpu').detach().numpy()
                fig = plt.figure(num=1, figsize=(10,10))
                for j in range(img_index**2):
                    fig.add_subplot(img_index,img_index,j+1)
                    plt.imshow(pixels[j,:], cmap='gray')
                    plt.axis('off')
                writer.add_figure(f'train_viz/real_input_disc', fig, epoch)
                pixels = torch.squeeze(torch.unflatten(gen_corrupted, 1, sizes), 1).to('cpu').detach().numpy()
                fig = plt.figure(num=1, figsize=(10,10))
                for j in range(img_index**2):
                    fig.add_subplot(img_index,img_index,j+1)
                    plt.imshow(pixels[j,:], cmap='gray')
                    plt.axis('off')
                writer.add_figure(f'train_viz/fake_input_disc', fig, epoch)
        if batch % 50 == 0:
            gen_loss, disc_loss, current = gen_loss.item(), disc_loss.item(), (batch + 1) * len(both_loader)
            print(f"gen_loss: {gen_loss:>7f}, disc_loss: {disc_loss} [{current:>5d}/{size:>5d}]")

def test(na_dataloader, nona_dataloader, model, dcon, mcon, tcon, writer=None, epoch=-1):
    """
    Testing loop for the Mask Predictor (MP) model. This function is called after each epoch of training.

    Parameters
    ----------
    na_dataloader: torch.utils.data.DataLoader
        Dataloader for testing data with missingness.
    nona_dataloader: torch.utils.data.DataLoader
        Dataloader for testing data without missingness.
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
        for i, (X, y, z, w) in enumerate(na_dataloader):
            random_generator = torch.Generator(device=tcon['device']).manual_seed(0)

            # set up hooks for saving activations
            hooks = []
            if i == 0 and image and tcon['activations'] and log:
                layer_iterator = enumerate(model.generator)
                for (k, layer) in layer_iterator:
                    if k % 2 != 0:
                        hooks.append(layer.register_forward_hook(getActivation(f'activations_layer_{k}', writer, img_index, epoch)))

            # X: actual corrupted image, y: mask, z: uncorrupted image, w:label
            X = X.to(tcon['device'])
            z = z.to(tcon['device'])

            noise = torch.randn((y.size(0), mcon['layer_dims_gen'][0]), device=tcon['device'], generator=random_generator)
            

            gen_mask_probs = model.generator_forward(noise)
            gen_mask = torch.bernoulli(gen_mask_probs, generator=random_generator).to(tcon['device']).bool()
            
            gen_corrupted_img = z.clone()
            gen_corrupted_img[gen_mask] = dcon['replacement']

            # write an example output, some weights and activations to tensorboard
            if i == 0 and image and log:
                if epoch == 1 or epoch == -1:
                    pixels = torch.squeeze(torch.unflatten(X, 1, sizes), 1).to('cpu').detach().numpy()
                    fig = plt.figure(num=1, figsize=(10,10))
                    for j in range(img_index**2):
                        fig.add_subplot(img_index,img_index,j+1)
                        plt.imshow(pixels[j,:], cmap='gray')
                        plt.axis('off')
                    writer.add_figure(f'na_sample/real_missingness', fig, epoch)

                pixels = torch.squeeze(torch.unflatten(gen_corrupted_img, 1, sizes), 1).to('cpu').detach().numpy()
                fig = plt.figure(num=1, figsize=(10,10))
                for j in range(img_index**2):
                    fig.add_subplot(img_index,img_index,j+1)
                    plt.imshow(pixels[j,:], cmap='gray')
                    plt.axis('off')
                writer.add_figure(f'na_sample/generated_mask', fig, epoch)

                pixels = torch.squeeze(torch.unflatten(gen_mask_probs, 1, sizes), 1).to('cpu').detach().numpy()
                fig = plt.figure(num=1, figsize=(10,10))
                for j in range(img_index**2):
                    fig.add_subplot(img_index,img_index,j+1)
                    plt.imshow(pixels[j,:], cmap='gray')
                    plt.axis('off')
                writer.add_figure(f'na_sample/generated_mask_probs', fig, epoch)


                viz_weights_to_board(model, writer, epoch=epoch, subplot_number=1)
                for h in hooks:
                    h.remove()
            break
    


def train_model(model, optimizer_gen, optimizer_disc, scheduler_gen, scheduler_disc,
                           dcon, mcon, tcon,
                           train_na_dataloader, train_nona_dataloader, train_both_loader,
                           test_na_dataloader, test_nona_dataloader):
    """
    Full training method for a GAN as a Mask Predictor (MP) model.

    Parameters:
    model: nn.Module
        The model to be trained.
    loss_fn: nn.Module
        The loss function to be used for training.
    optimizer_gen: torch.optim
        The optimizer to be used for training the generator.
    optimizer_disc: torch.optim
        The optimizer to be used for training the discriminator.
    scheduler_gen: torch.optim.lr_scheduler
        The learning rate scheduler to be used for training the generator.
    scheduler_disc: torch.optim.lr_scheduler
        The learning rate scheduler to be used for training the discriminator.
    dcon: dict
        Dictionary containing the data configuration.
    mcon: dict
        Dictionary containing the model configuration.
    tcon: dict
        Dictionary containing the training configuration.
    train_na_dataloader: torch.utils.data.DataLoader
        Dataloader for training data with missingness.
    train_nona_dataloader: torch.utils.data.DataLoader
        Dataloader for training data without missingness.
    train_both_loader: torch.utils.data.DataLoader
        Dataloader for training data with and without missingness. 
        Is a dataloader that returns samples of both simultaneously to train generator and discriminator with the same batch. Can be created with the DatasetZipped class.
    test_na_dataloader: torch.utils.data.DataLoader
        Dataloader for testing data with missingness.
    test_nona_dataloader: torch.utils.data.DataLoader
        Dataloader for testing data without missingness.
    """
    if not tcon['new_training']:
        return
    time = datetime.now().strftime("%B%d_%H_%M")
    writer = None
    random_generator = torch.Generator(device=tcon['device']).manual_seed(0)
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
        train(na_dataloader=train_na_dataloader, nona_dataloader=train_nona_dataloader, both_loader=train_both_loader, model=model, optimizer_disc=optimizer_disc, optimizer_gen=optimizer_gen, writer=writer, 
              dcon=dcon, mcon=mcon, tcon=tcon, epoch=t, random_generator=random_generator)
        test(na_dataloader=test_na_dataloader, nona_dataloader=test_nona_dataloader, model=model, 
             dcon=dcon, mcon=mcon, tcon=tcon, writer=writer, epoch=t)
        scheduler_gen.step() if mcon['lr_decay'] else None
        scheduler_disc.step() if mcon['lr_decay'] else None
        if writer is not None:
            writer.add_scalar("LR_gen", scheduler_gen.get_last_lr()[0], t)
            writer.add_scalar("LR_disc", scheduler_disc.get_last_lr()[0], t)
    print("Done!")

    if tcon['save_model']:
        torch.save({'model_state_dict': model.state_dict(), 
                    'optimizer_gen_state_dict': optimizer_gen.state_dict(),
                    'optimizer_disc_state_dict': optimizer_disc.state_dict(),
                    'scheduler_gen_state_dict': scheduler_gen.state_dict(),
                    'scheduler_disc_state_dict': scheduler_disc.state_dict(),
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


    