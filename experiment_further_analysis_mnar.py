import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import importlib
from sklearn.impute import SimpleImputer
import utils
import data
import modelMP
import modelSDAE
import helper_train_MP
import helper_train_SDAE
import helper_train_MP_GAN
import helper_noise

importlib.reload(utils)
importlib.reload(data)
importlib.reload(modelMP)
importlib.reload(helper_train_MP)
importlib.reload(modelSDAE)
importlib.reload(helper_train_SDAE)
importlib.reload(helper_noise)
importlib.reload(helper_train_MP_GAN)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
torch.random.manual_seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# randomness for the training of the Benchmark DAE
def dae_noise(x):
    return torch.rand_like(x)

# some overall experiment parameters
dataset = 'FashionMNIST'
noise_mechanism = 'mnar'
EPOCHS = 10
start_seed = 0
mse_all_noise_level = torch.zeros(3, 4)
mse_all_na_obs_perc = torch.zeros(3, 4)
folder_path = 'results/further_analysis/sensitivity'

os.makedirs(folder_path, exist_ok=True)

for i, noise_level in enumerate([0.1, 0.3, 0.5, 0.7]):
            print('Noise level:', noise_level)
            torch.random.manual_seed(1)
            dcon = {
                'dataset': dataset, # one of 'MNIST', 'FashionMNIST', 'CIFAR10'
                'noise_mechanism': noise_mechanism, # missingness mechanism
                'na_obs_percentage': 0.4, # the number of observations that have missing values
                'replacement': 'uniform', # what value is plugged in for missing values in the observations with missing values (number or 'uniform')
                'noise_level': noise_level, # the percentage of missing values per observation (share of features that are missing)
                'download': False,
                'regenerate': True,
                'device': device,


                # for noise_mechanism 'patches'
                'patch_size_axis': 5, # size of the patch in the x and y direction

                # for noise_mechanism MAR and MNAR
                'randperm_cols': False, # whether to shuffle the columns of the data matrix before applying the noise mechanism
                'average_missing_rates': 'normal', # can be 'uniform', 'normal' or a list/tuple/tensor of length no. of features, determines how the missingness rates are generated
                # if uniform, noise_level is used as mean for uniform distribution and impossible values are clipped to the interval [0,1] --> noise_level != exact missingness rate in mask
                # if normal, noise_level is ignored and the options below apply
                'chol_eps': 1e-6, # epsilon for the cholesky decomposition (epsilon*I is added to the covariance matrix to make it positive definite)
                'sigmoid_offset': [0.15, 0.0, -0.15, -0.25][i], # offset for the sigmoid function applied to the average missing rates generated by MultivariateNormal
                'sigmoid_k': 10, # steepness of sigmoid

                # for MNAR only
                'dependence': 'simple_unobserved', # can be 'simple_unobserved', 'complex_unobserved', 'unobserved_and_observed'


            }
            torch.random.manual_seed(1)
            data_without_nas_1 = data.ImputationDatasetGen(config=dcon, missing_vals=False)
            data_with_nas_1 = data.ImputationDatasetGen(config=dcon, missing_vals=True)

            ## Encoder model
            mcon0 = {
                'architecture': 'encoder_model_dae',
                'loss': 'full', # must be full otherwise error
                'epochs': EPOCHS,
                'batch_size': 64,
                'learning_rate': 3e-4,
                'lr_decay': False,
                'gamma': 2e-4,
                'step_size': 45,
                'layer_dims_enc': [784, 2000, 700],
                'layer_dims_dec': [700, 2000, 784],
                'device': device,
                'relu': True,
                'image': True,
                'noise_model': dae_noise,
                'corruption_share': noise_level, # level of the dropout noise that is used for training the DAE
                'mask_between_epochs': 'equal', # equal or random, determines the scope of the random generator that is passed to the mask bernoulli sampling function
                'additional_noise': 0, # does not apply here
            }

            tcon0 = {
                'new_training': 1,
                'log': 0,
                'save_model': 0,
                'img_index': 4, # index of the image to be plotted
                'activations': 0,
                'device': device,
                'train_val_test_split': [0.8, 0.2, 0]
            }

            if 'MNIST' in dcon['dataset']:
                mcon0['layer_dims_enc'][0] = 784
                mcon0['layer_dims_dec'][-1] = 784
            elif 'CIFAR10' in dcon['dataset']:
                mcon0['layer_dims_enc'][0] = 1024
                mcon0['layer_dims_dec'][-1] = 1024

                
            nona_train_loader_0 = DataLoader(data.DatasetWithSplits(data_without_nas_1, 'train', tcon0['train_val_test_split']), batch_size=mcon0['batch_size'], shuffle=True)
            nona_val_loader_0 = DataLoader(data.DatasetWithSplits(data_without_nas_1, 'validation', tcon0['train_val_test_split']), batch_size=mcon0['batch_size'], shuffle=False)

            model_autoencoder = modelSDAE.SyntheticDenoisingAutoEncoder(noise_model=dae_noise, layer_dims_enc=mcon0['layer_dims_enc'], layer_dims_dec=mcon0['layer_dims_dec'], relu=mcon0['relu'], image=mcon0['image']).to(device)
            loss_fn_autoencoder = nn.MSELoss(reduction='none')
            optimizer_autoencoder = torch.optim.Adam(model_autoencoder.parameters(), lr=mcon0['learning_rate'])
            scheduler_autoencoder = StepLR(optimizer_autoencoder, step_size=mcon0['step_size'], gamma=mcon0['gamma'])
            helper_train_SDAE.train_imputation_model(model=model_autoencoder, encoder=None, loss_fn=loss_fn_autoencoder, optimizer=optimizer_autoencoder, scheduler=scheduler_autoencoder,
                                                dcon=dcon, mcon=mcon0, tcon=tcon0,
                                                train_dataloader=nona_train_loader_0, validation_dataloader=nona_val_loader_0,
                                                noise_model=dae_noise)
            ## MP model
            model_autoencoder.eval()
            mcon = {
                'architecture': 'mask_pred_mlp',
                'epochs': EPOCHS,
                'batch_size': 64,
                'learning_rate': 3e-4,
                'lr_decay': False,
                'gamma': 2e-4,
                'step_size': 45,
                'layer_dims': [784, 2000, 2000, 2000, 784],
                'dropout': 0.5,
                'device': device,
                'relu': True,
                'image': True,
                'encoder': str(model_autoencoder)
                
            }

            tcon = {
                'new_training': 1,
                'log': 0,
                'save_model': 0,
                'img_index': 4, # index of the image to be plotted
                'activations': 0,
                'device': device,
                'train_val_test_split': [0.8, 0.2, 0]
            }

            if 'MNIST' in dcon['dataset']:
                mcon['layer_dims'][0] = mcon0['layer_dims_enc'][-1]
                mcon['layer_dims'][-1] = 784

            elif 'CIFAR10' in dcon['dataset']:
                mcon['layer_dims'][0] = mcon0['layer_dims_enc'][-1]
                mcon['layer_dims'][-1] = 1024



            nona_train_loader_1 = DataLoader(data.DatasetWithSplits(data_without_nas_1, 'train', tcon['train_val_test_split']), batch_size=mcon['batch_size'], shuffle=True)
            nona_val_loader_1 = DataLoader(data.DatasetWithSplits(data_without_nas_1, 'validation', tcon['train_val_test_split']), batch_size=mcon['batch_size'], shuffle=False)
            nona_test_loader_1 = DataLoader(data.DatasetWithSplits(data_without_nas_1, 'test', tcon['train_val_test_split']), batch_size=mcon['batch_size'], shuffle=False)

            na_train_loader_1 = DataLoader(data.DatasetWithSplits(data_with_nas_1, 'train', tcon['train_val_test_split']), batch_size=mcon['batch_size'], shuffle=True)
            na_val_loader_1 = DataLoader(data.DatasetWithSplits(data_with_nas_1, 'validation', tcon['train_val_test_split']), batch_size=mcon['batch_size'], shuffle=False)
            na_test_loader_1 = DataLoader(data.DatasetWithSplits(data_with_nas_1, 'test', tcon['train_val_test_split']), batch_size=mcon['batch_size'], shuffle=False)

            model_mp = modelMP.MaskPredMLP(layer_dims=mcon['layer_dims'], dropout=mcon['dropout'], relu=mcon['relu'], image=mcon['image']).to(device)



            loss_fn = nn.BCELoss()
            optimizer = torch.optim.Adam(model_mp.parameters(), lr=mcon['learning_rate'])
            scheduler = StepLR(optimizer, step_size=mcon['step_size'], gamma=mcon['gamma'])

            # train model
            helper_train_MP.train_model(model=model_mp, encoder=model_autoencoder.encoder, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                                                dcon=dcon, mcon=mcon, tcon=tcon,
                                                train_dataloader=na_train_loader_1, validation_dataloader=na_val_loader_1
                                                )
            helper_train_MP.test(dataloader=na_val_loader_1, model=model_mp, encoder=model_autoencoder.encoder, loss_fn=loss_fn, tcon=tcon, dcon=dcon, mcon=mcon)
            
            ## Define Synthetic Denoising Autoencoder
            model_mp.eval()
            model_autoencoder.eval()
            dcon2 = dcon.copy()
            dcon2['replacement'] = 0
            dcon2['regenerate'] = True

            mcon2 = {
                'architecture': 'synthetic_dae',
                'loss': 'full', # full or focused
                'epochs': EPOCHS,
                'batch_size': 64,
                'learning_rate': 3e-4,
                'lr_decay': False,
                'gamma': 2e-4,
                'step_size': 45,
                'layer_dims_enc': [784, 2000, 2000, 2000],
                'layer_dims_dec': [2000, 2000, 2000, 784],
                'device': device,
                'relu': True,
                'image': True,
                'noise_model': str(model_mp),
                'encoder': str(model_autoencoder),
                'corruption_share': -1, # the share of features that are corrupted in the training of the DAE
                'mask_between_epochs': 'equal', # equal or random, determines the scope of the random generator that is passed to the mask bernoulli sampling function
                'additional_noise': 0, # the share of additional noise that is added to the data during training

            }
            if 'MNIST' in dcon2['dataset']:
                mcon2['layer_dims_enc'][0] = 784
                mcon2['layer_dims_dec'][-1] = 784
            elif 'CIFAR10' in dcon2['dataset']:
                mcon2['layer_dims_enc'][0] = 1024
                mcon2['layer_dims_dec'][-1] = 1024


            tcon2 = {
                'new_training': 1,
                'log': 0,
                'save_model': 0,
                'img_index': 10, # index of the image to be plotted
                'activations': 0,
                'device': device,
                'train_val_test_split': [0.8, 0.2, 0]
            }

            torch.random.manual_seed(1)

            data_without_nas_2 = data.ImputationDatasetGen(config=dcon2, missing_vals=False)
            data_with_nas_2 = data.ImputationDatasetGen(config=dcon2, missing_vals=True)

            nona_train_loader_2 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'train', tcon2['train_val_test_split']), batch_size=mcon2['batch_size'], shuffle=True)
            nona_val_loader_2 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'validation', tcon2['train_val_test_split']), batch_size=mcon2['batch_size'], shuffle=False)
            nona_test_loader_2 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'test', tcon2['train_val_test_split']), batch_size=mcon2['batch_size'], shuffle=False)

            na_test_loader_2 = DataLoader(data.DatasetWithSplits(data_with_nas_2, 'test', [0, 0, 1]), batch_size=mcon2['batch_size'], shuffle=False) # here shuffle false, because it is only used for testing
            model_sdae = modelSDAE.SyntheticDenoisingAutoEncoder(noise_model=model_mp, layer_dims_enc=mcon2['layer_dims_enc'], layer_dims_dec=mcon2['layer_dims_dec'], relu=mcon2['relu'], image=mcon2['image']).to(device)
            loss_fn_sdae = nn.MSELoss(reduction='none')
            optimizer_sdae = torch.optim.Adam(model_sdae.parameters(), lr=mcon2['learning_rate'])
            scheduler_sdae = StepLR(optimizer_sdae, step_size=mcon2['step_size'], gamma=mcon2['gamma'])
            helper_train_SDAE.train_imputation_model(model=model_sdae, encoder=model_autoencoder.encoder, loss_fn=loss_fn_sdae, optimizer=optimizer_sdae, scheduler=scheduler_sdae,
                                                dcon=dcon2, mcon=mcon2, tcon=tcon2,
                                                train_dataloader=nona_train_loader_2, validation_dataloader=nona_val_loader_2, test_dataloader=na_test_loader_2,
                                                noise_model=model_mp)
            mse_all_noise_level[0, i] = helper_train_SDAE.test(dataloader=na_test_loader_2, model=model_sdae, loss_fn=loss_fn_sdae, dcon=dcon2, mcon=mcon2, tcon=tcon2)

            ## Benchmark Models
            ### Benchmark DAE

            dcon3 = dcon2.copy()

            mcon3 = {
                'architecture': 'benchmark_dae',
                'loss': 'full', # full or focused
                'epochs': EPOCHS,
                'batch_size': 64,
                'learning_rate': 3e-4,
                'lr_decay': False,
                'gamma': 2e-4,
                'step_size': 45,
                'layer_dims_enc': [784, 2000, 2000],
                'layer_dims_dec': [2000, 2000, 784],
                'device': device,
                'relu': True,
                'image': True,
                'noise_model': dae_noise,
                'corruption_share': noise_level, # the share of features that are corrupted in the training of the DAE
                'mask_between_epochs': 'random', # (DOES NOT APPLY to benchmark_DAE)
                'additional_noise': 0, # the share of additional noise that is added to the data during training
            }
            if 'MNIST' in dcon3['dataset']:
                mcon3['layer_dims_enc'][0] = 784
                mcon3['layer_dims_dec'][-1] = 784
            elif 'CIFAR10' in dcon3['dataset']:
                mcon3['layer_dims_enc'][0] = 1024
                mcon3['layer_dims_dec'][-1] = 1024

            tcon3 = {
                'new_training': 1,
                'log': 0,
                'save_model': 0,
                'img_index': 10, # index of the image to be plotted
                'activations': 0,
                'device': device,
                'train_val_test_split': [0.8, 0.2, 0]
            }
            torch.random.manual_seed(1)

            nona_train_loader_3 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'train', tcon3['train_val_test_split']), batch_size=mcon3['batch_size'], shuffle=True)
            nona_val_loader_3 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'validation', tcon3['train_val_test_split']), batch_size=mcon3['batch_size'], shuffle=False)
            nona_test_loader_3 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'test', tcon3['train_val_test_split']), batch_size=mcon3['batch_size'], shuffle=False)

            na_test_loader_3 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'test', [0, 0, 1]), batch_size=mcon3['batch_size'], shuffle=False) # here shuffle false, because it is only used for testing
            model_bdae = modelSDAE.SyntheticDenoisingAutoEncoder(noise_model=dae_noise, layer_dims_enc=mcon3['layer_dims_enc'], layer_dims_dec=mcon3['layer_dims_dec'], relu=mcon3['relu'], image=mcon3['image']).to(device)
            loss_fn_bdae = nn.MSELoss(reduction='none')
            optimizer_bdae = torch.optim.Adam(model_bdae.parameters(), lr=mcon3['learning_rate'])
            scheduler_bdae = StepLR(optimizer_bdae, step_size=mcon3['step_size'], gamma=mcon3['gamma'])
            helper_train_SDAE.train_imputation_model(model=model_bdae, encoder=None, loss_fn=loss_fn_bdae, optimizer=optimizer_bdae, scheduler=scheduler_bdae,
                                                dcon=dcon3, mcon=mcon3, tcon=tcon3,
                                                train_dataloader=nona_train_loader_3, validation_dataloader=nona_val_loader_3, test_dataloader=na_test_loader_3,
                                                noise_model=dae_noise)
            mse_all_noise_level[1, i] = helper_train_SDAE.test(dataloader=na_test_loader_3, model=model_bdae, loss_fn=loss_fn_bdae, dcon=dcon3, mcon=mcon3, tcon=tcon3)

            # Mean imputation
            data_with_nas_as_nas = data_with_nas_2.data.clone()
            mask = data_with_nas_2.targets.clone()
            data_with_nas_as_nas[mask == 1] = float('nan')
            full_data = torch.cat((data_without_nas_2.data.cpu(), data_with_nas_as_nas.cpu()), dim=0).detach()
            full_targets = torch.cat((data_without_nas_2.labels.cpu(), data_with_nas_2.labels.cpu()), dim=0).detach()

            full_data_imputed = torch.tensor(SimpleImputer().fit_transform(full_data)).float().to(device)
            ground_truth = data_with_nas_2.unmissing_data
            imputed_data = full_data_imputed[range(data_with_nas_as_nas.size(0)), :]

            mse = torch.sum(nn.MSELoss(reduction='none')(imputed_data.cpu(), ground_truth.cpu()) * (mask.cpu() == 1).float()) / torch.sum(mask.cpu())
            mse_all_noise_level[2, i] = mse


# save results to csv
mse_this_run_df = pd.DataFrame(mse_all_noise_level, 
                                columns=['MNAR_10', 'MNAR_30', 'MNAR_50', 'MNAR_70'],
                                index=['SDAE', 'DAE', 'Mean'])

mse_this_run_df.to_csv(f'{folder_path}/mse_noise_level.csv')


##################################################################
############### NA_OBS_PERCENTAGE ################################
##################################################################
for i, na_obs_perc in enumerate([0.1, 0.3, 0.5, 0.7]):
            print('NA obs percentage:', na_obs_perc)
            torch.random.manual_seed(1)
            dcon = {
                'dataset': dataset, #one of 'MNIST', 'FashionMNIST', 'CIFAR10'
                'noise_mechanism': noise_mechanism, # missingness mechanism
                'na_obs_percentage': na_obs_perc, # the number of observations that have missing values
                'replacement': 'uniform', # what value is plugged in for missing values in the observations with missing values (number or 'uniform')
                'noise_level': 0.2, # the percentage of missing values per observation (share of features that are missing)
                'download': False,
                'regenerate': True,
                'device': device,


                # for noise_mechanism 'patches'
                'patch_size_axis': 5, # size of the patch in the x and y direction

                # for noise_mechanism MAR and MNAR
                'randperm_cols': False, # whether to shuffle the columns of the data matrix before applying the noise mechanism
                'average_missing_rates': 'normal', # can be 'uniform', 'normal' or a list/tuple/tensor of length no. of features, determines how the missingness rates are generated
                # if uniform, noise_level is used as mean for uniform distribution and impossible values are clipped to the interval [0,1] --> noise_level != exact missingness rate in mask
                # if normal, noise_level is ignored and the options below apply
                'chol_eps': 1e-6, # epsilon for the cholesky decomposition (epsilon*I is added to the covariance matrix to make it positive definite)
                'sigmoid_offset': 0.05, # offset for the sigmoid function applied to the average missing rates generated by MultivariateNormal
                'sigmoid_k': 10, # steepness of sigmoid

                # for MNAR only
                'dependence': 'simple_unobserved', # can be 'simple_unobserved', 'complex_unobserved', 'unobserved_and_observed'


            }
            torch.random.manual_seed(1)
            data_without_nas_1 = data.ImputationDatasetGen(config=dcon, missing_vals=False)
            data_with_nas_1 = data.ImputationDatasetGen(config=dcon, missing_vals=True)
            ## Encoder model
            mcon0 = {
                'architecture': 'encoder_model_dae',
                'loss': 'full', # must be full otherwise error
                'epochs': EPOCHS,
                'batch_size': 64,
                'learning_rate': 3e-4,
                'lr_decay': False,
                'gamma': 2e-4,
                'step_size': 45,
                'layer_dims_enc': [784, 2000, 700],
                'layer_dims_dec': [700, 2000, 784],
                'device': device,
                'relu': True,
                'image': True,
                'noise_model': dae_noise,
                'corruption_share': 0.2, # level of the masking noise that is used for training the DAE
                'mask_between_epochs': 'equal', # equal or random, determines the scope of the random generator that is passed to the mask bernoulli sampling function
                'additional_noise': 0, # does not apply here
            }

            tcon0 = {
                'new_training': 1,
                'log': 0,
                'save_model': 0,
                'img_index': 4, # index of the image to be plotted
                'activations': 0,
                'device': device,
                'train_val_test_split': [0.8, 0.2, 0]
            }

            if 'MNIST' in dcon['dataset']:
                mcon0['layer_dims_enc'][0] = 784
                mcon0['layer_dims_dec'][-1] = 784
            elif 'CIFAR10' in dcon['dataset']:
                mcon0['layer_dims_enc'][0] = 1024
                mcon0['layer_dims_dec'][-1] = 1024

                
            nona_train_loader_0 = DataLoader(data.DatasetWithSplits(data_without_nas_1, 'train', tcon0['train_val_test_split']), batch_size=mcon0['batch_size'], shuffle=True)
            nona_val_loader_0 = DataLoader(data.DatasetWithSplits(data_without_nas_1, 'validation', tcon0['train_val_test_split']), batch_size=mcon0['batch_size'], shuffle=False)

            model_autoencoder = modelSDAE.SyntheticDenoisingAutoEncoder(noise_model=dae_noise, layer_dims_enc=mcon0['layer_dims_enc'], layer_dims_dec=mcon0['layer_dims_dec'], relu=mcon0['relu'], image=mcon0['image']).to(device)
            loss_fn_autoencoder = nn.MSELoss(reduction='none')
            optimizer_autoencoder = torch.optim.Adam(model_autoencoder.parameters(), lr=mcon0['learning_rate'])
            scheduler_autoencoder = StepLR(optimizer_autoencoder, step_size=mcon0['step_size'], gamma=mcon0['gamma'])
            helper_train_SDAE.train_imputation_model(model=model_autoencoder, encoder=None, loss_fn=loss_fn_autoencoder, optimizer=optimizer_autoencoder, scheduler=scheduler_autoencoder,
                                                dcon=dcon, mcon=mcon0, tcon=tcon0,
                                                train_dataloader=nona_train_loader_0, validation_dataloader=nona_val_loader_0,
                                                noise_model=dae_noise)
            ## MP model
            model_autoencoder.eval()
            mcon = {
                'architecture': 'mask_pred_mlp',
                'epochs': EPOCHS,
                'batch_size': 64,
                'learning_rate': 3e-4,
                'lr_decay': False,
                'gamma': 2e-4,
                'step_size': 45,
                'layer_dims': [784, 2000, 2000, 2000, 784],
                'dropout': 0.5,
                'device': device,
                'relu': True,
                'image': True,
                'encoder': str(model_autoencoder)
                
            }

            tcon = {
                'new_training': 1,
                'log': 0,
                'save_model': 0,
                'img_index': 4, # index of the image to be plotted
                'activations': 0,
                'device': device,
                'train_val_test_split': [0.8, 0.2, 0]
            }

            if 'MNIST' in dcon['dataset']:
                mcon['layer_dims'][0] = mcon0['layer_dims_enc'][-1]
                mcon['layer_dims'][-1] = 784

            elif 'CIFAR10' in dcon['dataset']:
                mcon['layer_dims'][0] = mcon0['layer_dims_enc'][-1]
                mcon['layer_dims'][-1] = 1024



            nona_train_loader_1 = DataLoader(data.DatasetWithSplits(data_without_nas_1, 'train', tcon['train_val_test_split']), batch_size=mcon['batch_size'], shuffle=True)
            nona_val_loader_1 = DataLoader(data.DatasetWithSplits(data_without_nas_1, 'validation', tcon['train_val_test_split']), batch_size=mcon['batch_size'], shuffle=False)
            nona_test_loader_1 = DataLoader(data.DatasetWithSplits(data_without_nas_1, 'test', tcon['train_val_test_split']), batch_size=mcon['batch_size'], shuffle=False)

            na_train_loader_1 = DataLoader(data.DatasetWithSplits(data_with_nas_1, 'train', tcon['train_val_test_split']), batch_size=mcon['batch_size'], shuffle=True)
            na_val_loader_1 = DataLoader(data.DatasetWithSplits(data_with_nas_1, 'validation', tcon['train_val_test_split']), batch_size=mcon['batch_size'], shuffle=False)
            na_test_loader_1 = DataLoader(data.DatasetWithSplits(data_with_nas_1, 'test', tcon['train_val_test_split']), batch_size=mcon['batch_size'], shuffle=False)

            model_mp = modelMP.MaskPredMLP(layer_dims=mcon['layer_dims'], dropout=mcon['dropout'], relu=mcon['relu'], image=mcon['image']).to(device)

            loss_fn = nn.BCELoss()
            optimizer = torch.optim.Adam(model_mp.parameters(), lr=mcon['learning_rate'])
            scheduler = StepLR(optimizer, step_size=mcon['step_size'], gamma=mcon['gamma'])

            # train model
            helper_train_MP.train_model(model=model_mp, encoder=model_autoencoder.encoder, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                                                dcon=dcon, mcon=mcon, tcon=tcon,
                                                train_dataloader=na_train_loader_1, validation_dataloader=na_val_loader_1
                                                )
            helper_train_MP.test(dataloader=na_val_loader_1, model=model_mp, encoder=model_autoencoder.encoder, loss_fn=loss_fn, tcon=tcon, dcon=dcon, mcon=mcon)
            
            ## Define Synthetic Denoising Autoencoder
            model_mp.eval()
            model_autoencoder.eval()
            dcon2 = dcon.copy()
            dcon2['replacement'] = 0
            dcon2['regenerate'] = True

            mcon2 = {
                'architecture': 'synthetic_dae',
                'loss': 'full', # full or focused
                'epochs': EPOCHS,
                'batch_size': 64,
                'learning_rate': 3e-4,
                'lr_decay': False,
                'gamma': 2e-4,
                'step_size': 45,
                'layer_dims_enc': [784, 2000, 2000, 2000],
                'layer_dims_dec': [2000, 2000, 2000, 784],
                'device': device,
                'relu': True,
                'image': True,
                'noise_model': str(model_mp),
                'encoder': str(model_autoencoder),
                'corruption_share': -1, # the share of features that are corrupted in the training of the DAE
                'mask_between_epochs': 'equal', # equal or random, determines the scope of the random generator that is passed to the mask bernoulli sampling function
                'additional_noise': 0, # the share of additional noise that is added to the data during training

            }
            if 'MNIST' in dcon2['dataset']:
                mcon2['layer_dims_enc'][0] = 784
                mcon2['layer_dims_dec'][-1] = 784
            elif 'CIFAR10' in dcon2['dataset']:
                mcon2['layer_dims_enc'][0] = 1024
                mcon2['layer_dims_dec'][-1] = 1024


            tcon2 = {
                'new_training': 1,
                'log': 0,
                'save_model': 0,
                'img_index': 10, # index of the image to be plotted
                'activations': 0,
                'device': device,
                'train_val_test_split': [0.8, 0.2, 0]
            }

            torch.random.manual_seed(1)

            data_without_nas_2 = data.ImputationDatasetGen(config=dcon2, missing_vals=False)
            data_with_nas_2 = data.ImputationDatasetGen(config=dcon2, missing_vals=True)

            nona_train_loader_2 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'train', tcon2['train_val_test_split']), batch_size=mcon2['batch_size'], shuffle=True)
            nona_val_loader_2 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'validation', tcon2['train_val_test_split']), batch_size=mcon2['batch_size'], shuffle=False)
            nona_test_loader_2 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'test', tcon2['train_val_test_split']), batch_size=mcon2['batch_size'], shuffle=False)

            na_test_loader_2 = DataLoader(data.DatasetWithSplits(data_with_nas_2, 'test', [0, 0, 1]), batch_size=mcon2['batch_size'], shuffle=False) # here shuffle false, because it is only used for testing
            model_sdae = modelSDAE.SyntheticDenoisingAutoEncoder(noise_model=model_mp, layer_dims_enc=mcon2['layer_dims_enc'], layer_dims_dec=mcon2['layer_dims_dec'], relu=mcon2['relu'], image=mcon2['image']).to(device)
            loss_fn_sdae = nn.MSELoss(reduction='none')
            optimizer_sdae = torch.optim.Adam(model_sdae.parameters(), lr=mcon2['learning_rate'])
            scheduler_sdae = StepLR(optimizer_sdae, step_size=mcon2['step_size'], gamma=mcon2['gamma'])
            helper_train_SDAE.train_imputation_model(model=model_sdae, encoder=model_autoencoder.encoder, loss_fn=loss_fn_sdae, optimizer=optimizer_sdae, scheduler=scheduler_sdae,
                                                dcon=dcon2, mcon=mcon2, tcon=tcon2,
                                                train_dataloader=nona_train_loader_2, validation_dataloader=nona_val_loader_2, test_dataloader=na_test_loader_2,
                                                noise_model=model_mp)
            mse_all_na_obs_perc[0, i] = helper_train_SDAE.test(dataloader=na_test_loader_2, model=model_sdae, loss_fn=loss_fn_sdae, dcon=dcon2, mcon=mcon2, tcon=tcon2)

            ## Benchmark Models
            ### Benchmark DAE

            dcon3 = dcon2.copy()

            mcon3 = {
                'architecture': 'benchmark_dae',
                'loss': 'full', # full or focused
                'epochs': EPOCHS,
                'batch_size': 64,
                'learning_rate': 3e-4,
                'lr_decay': False,
                'gamma': 2e-4,
                'step_size': 45,
                'layer_dims_enc': [784, 2000, 2000],
                'layer_dims_dec': [2000, 2000, 784],
                'device': device,
                'relu': True,
                'image': True,
                'noise_model': dae_noise,
                'corruption_share': 0.2, # the share of features that are corrupted in the training of the DAE
                'mask_between_epochs': 'random', # (DOES NOT APPLY to benchmark_DAE)
                'additional_noise': 0, # the share of additional noise that is added to the data during training
            }
            if 'MNIST' in dcon3['dataset']:
                mcon3['layer_dims_enc'][0] = 784
                mcon3['layer_dims_dec'][-1] = 784
            elif 'CIFAR10' in dcon3['dataset']:
                mcon3['layer_dims_enc'][0] = 1024
                mcon3['layer_dims_dec'][-1] = 1024

            tcon3 = {
                'new_training': 1,
                'log': 0,
                'save_model': 0,
                'img_index': 10, # index of the image to be plotted
                'activations': 0,
                'device': device,
                'train_val_test_split': [0.8, 0.2, 0]
            }
            torch.random.manual_seed(1)

            nona_train_loader_3 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'train', tcon3['train_val_test_split']), batch_size=mcon3['batch_size'], shuffle=True)
            nona_val_loader_3 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'validation', tcon3['train_val_test_split']), batch_size=mcon3['batch_size'], shuffle=False)
            nona_test_loader_3 = DataLoader(data.DatasetWithSplits(data_without_nas_2, 'test', tcon3['train_val_test_split']), batch_size=mcon3['batch_size'], shuffle=False)

            na_test_loader_3 = DataLoader(data.DatasetWithSplits(data_with_nas_2, 'test', [0, 0, 1]), batch_size=mcon3['batch_size'], shuffle=False) # here shuffle false, because it is only used for testing
            model_bdae = modelSDAE.SyntheticDenoisingAutoEncoder(noise_model=dae_noise, layer_dims_enc=mcon3['layer_dims_enc'], layer_dims_dec=mcon3['layer_dims_dec'], relu=mcon3['relu'], image=mcon3['image']).to(device)
            loss_fn_bdae = nn.MSELoss(reduction='none')
            optimizer_bdae = torch.optim.Adam(model_bdae.parameters(), lr=mcon3['learning_rate'])
            scheduler_bdae = StepLR(optimizer_bdae, step_size=mcon3['step_size'], gamma=mcon3['gamma'])
            helper_train_SDAE.train_imputation_model(model=model_bdae, encoder=None, loss_fn=loss_fn_bdae, optimizer=optimizer_bdae, scheduler=scheduler_bdae,
                                                dcon=dcon3, mcon=mcon3, tcon=tcon3,
                                                train_dataloader=nona_train_loader_3, validation_dataloader=nona_val_loader_3, test_dataloader=na_test_loader_3,
                                                noise_model=dae_noise)
            mse_all_na_obs_perc[1, i] = helper_train_SDAE.test(dataloader=na_test_loader_3, model=model_bdae, loss_fn=loss_fn_bdae, dcon=dcon3, mcon=mcon3, tcon=tcon3)

            # Mean imputation
            data_with_nas_as_nas = data_with_nas_2.data.clone()
            mask = data_with_nas_2.targets.clone()
            data_with_nas_as_nas[mask == 1] = float('nan')
            full_data = torch.cat((data_without_nas_2.data.cpu(), data_with_nas_as_nas.cpu()), dim=0).detach()
            full_targets = torch.cat((data_without_nas_2.labels.cpu(), data_with_nas_2.labels.cpu()), dim=0).detach()

            full_data_imputed = torch.tensor(SimpleImputer().fit_transform(full_data)).float().to(device)
            ground_truth = data_with_nas_2.unmissing_data
            imputed_data = full_data_imputed[range(data_with_nas_as_nas.size(0)), :]

            mse = torch.sum(nn.MSELoss(reduction='none')(imputed_data.cpu(), ground_truth.cpu()) * (mask.cpu() == 1).float()) / torch.sum(mask.cpu())
            mse_all_na_obs_perc[2, i] = mse


# save results to csv
mse_this_run_df = pd.DataFrame(mse_all_na_obs_perc, 
                                columns=['NA_10', 'NA_30', 'NA_50', 'NA_70'],
                                index=['SDAE', 'DAE', 'Mean'])

mse_this_run_df.to_csv(f'{folder_path}/mse_na_obs_perc.csv')

# create plots
mse_na_obs_perc = pd.read_csv(f'{folder_path}/mse_na_obs_perc.csv', header=0, index_col=0)
rmse_na_obs_perc = np.sqrt(mse_na_obs_perc.values)
models = ['imputeLM', 'DAE', 'Mean']
plt.figure(figsize=(4, 3))
for i in range(rmse_na_obs_perc.shape[0]):
    plt.plot(rmse_na_obs_perc[i, :], label=models[i])
x_axis_values = [0.1, 0.3, 0.5, 0.7]
plt.xticks(range(4), x_axis_values)
plt.legend()
plt.xlabel('Proportion of incomplete observations')
plt.ylabel('RMSE')
plt.show()
plt.savefig(f'{folder_path}/mse_na_obs_perc.png')

mse_noise_level = pd.read_csv(f'{folder_path}/mse_noise_level.csv', header=0, index_col=0)
rmse_noise_level = np.sqrt(mse_noise_level.values)
models = ['imputeLM', 'DAE', 'Mean']
plt.figure(figsize=(4, 3))
for i in range(rmse_noise_level.shape[0]):
    plt.plot(rmse_noise_level[i, :], label=models[i])
x_axis_values = [0.1, 0.3, 0.5, 0.7]
plt.xticks(range(4), x_axis_values)
plt.legend()
plt.xlabel('Missing rate per observation')
plt.ylabel('RMSE')
plt.show()
plt.savefig(f'{folder_path}/mse_na_obs_perc.png')