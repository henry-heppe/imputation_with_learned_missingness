import torch
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import v2
import helper_noise
from datetime import datetime
import os

class ImputationDatasetGen(torch.utils.data.Dataset):
    """
    A class to generate datasets with missing values. Available datasets are MNIST, FashionMNIST, CIFAR10. The config dictionary contains an option whether the dataset should be downloaded (again).
    If missing_vals == True, getting one item of this dataset returns a tuple: partially-observed observation, missingness mask, uncorrupted observation, class label.
    If missing_vals == False, getting one item of this dataset returns a tuple: fully-observed observation, fully-observed observation, fully-observed observation, class label.

    Parameters
    ----------
    config : dict
        A dictionary containing the all parameters to generate the dataset.
    missing_vals : bool
        Whether to return the partially-observed set of observations (TRUE) or the fully-observed set (FALSE).
    input_data : torch.Tensor
        Optional: A tensor containing the data. Can be used to generate missing values on a specific dataset.

    """
    def __init__(self, config, missing_vals=False, input_data=None):
        # retrieve the dataset configuration settings
        self.name = config['dataset']
        self.noise_mechanism = config['noise_mechanism']
        self.na_obs_percentage = config['na_obs_percentage']
        self.replacement = config['replacement']
        self.noise_level = config['noise_level']
        self.download = config['download']        
        self.transform = config['transform'] if 'transform' in config else None
        self.target_transform = config['target_transform'] if 'target_transform' in config else None
        self.regenerate = config['regenerate']
        self.input_data = input_data

        self.missing_vals = missing_vals

        self.time = datetime.now().strftime("%B%d_%H_%M")
        
        # define the missingness mechanism
        if self.noise_mechanism == 'mcar':
            self.noise_adder_func = helper_noise.missingness_adder_mcar
        elif self.noise_mechanism == 'patch':
            self.noise_adder_func = helper_noise.missingness_adder_patch
        elif self.noise_mechanism == 'fixed_patch':
            self.noise_adder_func = helper_noise.missingness_adder_fixed_patch
        elif self.noise_mechanism == 'patches':
            self.noise_adder_func = helper_noise.missingness_adder_patches
        elif self.noise_mechanism == 'mar':
            self.noise_adder_func = helper_noise.missingness_adder_mar
        elif self.noise_mechanism == 'mnar':
            self.noise_adder_func = helper_noise.missingness_adder_mnar
        elif self.noise_mechanism == 'threshold':
            self.noise_adder_func = helper_noise.missingness_adder_threshold
        elif self.noise_mechanism == 'special_mar':
            self.noise_adder_func = helper_noise.MAR_mask
        elif self.noise_mechanism == 'special_mnar_log':
            self.noise_adder_func = helper_noise.MNAR_mask_logistic
        elif self.noise_mechanism == 'special_mnar_self_log':
            self.noise_adder_func = helper_noise.MNAR_self_mask_logistic
        elif self.noise_mechanism == 'special_mnar_quant':
            self.noise_adder_func = helper_noise.MNAR_mask_quantiles
        else:
            raise ValueError('noise_adder does not exist')

        # download/load the raw data  
        if self.name == 'MNIST':
            orig_training_data = datasets.MNIST(root="data", train=True, download=self.download)
            orig_test_data = datasets.MNIST(root="data", train=False, download=self.download)
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        elif self.name == 'FashionMNIST':
            orig_training_data = datasets.FashionMNIST(root="data", train=True, download=self.download)
            orig_test_data = datasets.FashionMNIST(root="data", train=False, download=self.download)
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        elif self.name == 'CIFAR10':
            orig_training_data = datasets.CIFAR10(root="data", train=True, download=self.download)
            orig_test_data = datasets.CIFAR10(root="data", train=False, download=self.download)
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Grayscale()])

            orig_training_data.data = torch.transpose(torch.transpose(torch.tensor(orig_training_data.data), 1, 3), 2, 3)
            orig_test_data.data = torch.transpose(torch.transpose(torch.tensor(orig_test_data.data), 1, 3), 2, 3)
            orig_training_data.targets = torch.tensor(orig_training_data.targets)
            orig_test_data.targets = torch.tensor(orig_test_data.targets)
        else:
            raise ValueError('dataset must be one of [MNIST, FashionMNIST, CIFAR10]')

        orig_training_data.data = transform(orig_training_data.data) if transform is not None else orig_training_data.data
        orig_test_data.data = transform(orig_test_data.data) if transform is not None else orig_test_data.data

        if self.input_data is not None:
            splits = torch.utils.data.random_split(self.input_data, [0.8, 0.2], torch.Generator().manual_seed(42))
            orig_training_data.data = self.input_data[splits[0].indices]
            orig_test_data.data = self.input_data[splits[1].indices]
            orig_training_data.targets = torch.zeros_like(orig_training_data.targets)
            orig_test_data.targets = torch.zeros_like(orig_test_data.targets)
            transform = None

        orig_training_data.data = torch.flatten(orig_training_data.data, start_dim=1, end_dim=-1)
        orig_test_data.data = torch.flatten(orig_test_data.data, start_dim=1, end_dim=-1)
        
        # if -1: use predefined split, else: split the data
        if self.na_obs_percentage == -1:
            if not missing_vals:
                self.data = orig_training_data.data
                self.targets = self.data.clone()
                self.labels = orig_training_data.targets
                return
                
            self.unmissing_data = orig_test_data.data
            self.labels = orig_test_data.targets

        else:    
            full_x = torch.cat([orig_training_data.data, orig_test_data.data], dim=0)
            full_y = torch.cat([orig_training_data.targets, orig_test_data.targets], dim=0)
            splits = torch.utils.data.random_split(full_x, [1-self.na_obs_percentage, self.na_obs_percentage], torch.Generator().manual_seed(42))

            if not missing_vals:
                self.data = full_x[splits[0].indices]
                self.targets = self.data.clone()
                self.labels = full_y[splits[0].indices]
                return
            
            self.unmissing_data = full_x[splits[1].indices]
            self.labels = full_y[splits[1].indices]

        # generate the missing values for the data or load a presaved corrupted dataset
        if self.regenerate:
            self.data, self.targets = self.noise_adder_func(dataset=self.unmissing_data, config=config)
            os.makedirs(f'data/{self.name}/processed/{self.noise_mechanism}/replacement{self.replacement}/na_obs_perc{int(abs(self.na_obs_percentage)*10)}/noise_level{int(self.noise_level*10)}', 
                        exist_ok=True)
            torch.save({'data': self.data, 'mask': self.targets, 'config': config},
                        f'data/{self.name}/processed/{self.noise_mechanism}/replacement{self.replacement}/na_obs_perc{int(abs(self.na_obs_percentage)*10)}/noise_level{int(self.noise_level*10)}/{self.time}.pt')
        else:
            last_dataset = sorted(os.listdir(f'data/{self.name}/processed/{self.noise_mechanism}/replacement{self.replacement}/na_obs_perc{int(abs(self.na_obs_percentage)*10)}/noise_level{int(self.noise_level*10)}'))[-1]
            last_dataset_path = os.path.join(f'data/{self.name}/processed/{self.noise_mechanism}/replacement{self.replacement}/na_obs_perc{int(abs(self.na_obs_percentage)*10)}/noise_level{int(self.noise_level*10)}', 
                                                       last_dataset)
            self.data_dict = torch.load(last_dataset_path)
            self.data = self.data_dict['data']
            self.targets = self.data_dict['mask']  
            self.config_loaded = self.data_dict['config']      

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, index):
        if self.transform:
            self.data = self.transform(self.data[index])
        if self.target_transform:
            self.targets = self.target_transform(self.targets[index])
        if self.missing_vals:
            # data is a corrupted image, target is the missingness mask, unmissing_data is the uncorrupted image and label is the image class
            return self.data[index], self.targets[index], self.unmissing_data[index], self.labels[index]
        else:
            # here the latter two can be ignored (as if they were None), they are just there so that the dataloader works and returns 4 elements
            return self.data[index], self.targets[index], self.data[index], self.labels[index]
    

class DatasetWithSplits(torch.utils.data.Dataset):
    """
    A class to split a dataset into training, validation and test set as pytorch datasets.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to split.
    train_val_test : str
        Whether to return the training, validation or test set.
    splits : list
        The split ratios for the training, validation and test set.
    """
    def __init__(self, dataset, train_val_test='train', splits=[0.8, 0.2, 0]):
        self.dataset = dataset
        self.x = dataset.data
        self.y = dataset.targets
        self.z = dataset.unmissing_data if hasattr(dataset, 'unmissing_data') else dataset.data
        self.w = dataset.labels if hasattr(dataset, 'labels') else None
        
        self.train_val_test = train_val_test
        self.splits = splits

        self.splits = torch.utils.data.random_split(self.dataset, self.splits, torch.Generator().manual_seed(42))
        if train_val_test == 'train':
            self.x = self.x[self.splits[0].indices]
            self.y = self.y[self.splits[0].indices]
            self.z = self.z[self.splits[0].indices]
            self.w = self.w[self.splits[0].indices]
        elif train_val_test == 'validation':
            self.x = self.x[self.splits[1].indices]
            self.y = self.y[self.splits[1].indices]
            self.z = self.z[self.splits[1].indices]
            self.w = self.w[self.splits[1].indices]
        elif train_val_test == 'test':
            self.x = self.x[self.splits[2].indices]
            self.y = self.y[self.splits[2].indices]
            self.z = self.z[self.splits[2].indices]
            self.w = self.w[self.splits[2].indices]
        else:
            raise ValueError('train_val_test must be either train, validation or test')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index], self.w[index]
        
    def get_splits(self):
        return self.splits
    
class DatasetZipped(torch.utils.data.Dataset):
    """
    A class to zip two datasets together. If they are not of the same length, the shorter one is repeated until it has the same length as the longer one.

    Parameters
    ----------
    dataset1 : torch.utils.data.Dataset
        The first dataset to zip.
    dataset2 : torch.utils.data.Dataset
        The second dataset to zip.

    """
    def __init__(self, dataset_nona, dataset_na):
        self.dataset_nona = dataset_nona
        self.dataset_na = dataset_na
        self.generator = torch.Generator().manual_seed(42)

        if len(self.dataset_nona) > len(self.dataset_na):
            self.dataset_na = torch.utils.data.ConcatDataset([self.dataset_na, torch.utils.data.Subset(self.dataset_na, range(len(self.dataset_nona)-len(self.dataset_na)))])
        elif len(self.dataset_nona) < len(self.dataset_na):
            self.dataset_nona = torch.utils.data.ConcatDataset([self.dataset_nona, torch.utils.data.Subset(self.dataset_nona, range(len(self.dataset_na)-len(self.dataset_nona)))])

    def __len__(self):
        return len(self.dataset_nona)

    def __getitem__(self, index):
        return self.dataset_nona[index], self.dataset_na[index]


if __name__ == '__main__':
    None
         
        
         
