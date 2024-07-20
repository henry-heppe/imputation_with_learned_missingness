import torch
import numpy as np
from scipy import optimize



def missingness_adder_mcar(dataset, config):
        """
        Add missing values to a dataset using a MCAR mechanism.

        Parameters
        ----------
        dataset : torch.Tensor
            Dataset to which missing values will be added.

        config : dict
            Dictionary containing the parameters of the missingness mechanism. The following keys are expected:
            - noise_level: float
                Proportion of missing values to generate.
            - replacement: float/str
                Value to replace the missing values with. If 'uniform', missing values are replaced by uniform random values.

        Returns
        -------
        x_noisy : torch.Tensor
            Dataset with missing values added.

        mask : torch.Tensor
            Mask of missing values (1 if the value is missing, 0 otherwise).
            
        """
        noise_level = config['noise_level']
        replacement = config['replacement']
        if noise_level == 0:
            return dataset
        else:
            x_noisy = dataset.clone()
            x_noisy = torch.flatten(x_noisy, start_dim=1, end_dim=-1)
            random_matrix = torch.rand_like(input=x_noisy)
            number_vars_affected = int(noise_level * x_noisy.size()[1])
            top_quantile = torch.topk(input=random_matrix, k=number_vars_affected, dim=1, sorted=True)[0][:,-1:]
            mask = random_matrix >= top_quantile
            if replacement == 'uniform':
                replacement = torch.randn_like(x_noisy)
                x_noisy[mask] = replacement[mask]
            else:
                x_noisy[mask] = replacement
            return x_noisy, mask.float()
        
def missingness_adder_patch(dataset, config):
    """
        Add missing values to a dataset in shape of a horizontal patch. The vertical placement of the patch is random.

        Parameters
        ----------
        dataset : torch.Tensor
            Dataset to which missing values will be added.

        config : dict
            Dictionary containing the parameters of the missingness mechanism. The following keys are expected:
            - noise_level: float
                Proportion of missing values to generate.
            - replacement: float/str
                Value to replace the missing values with. If 'uniform', missing values are replaced by uniform random values.

        Returns
        -------
        x_noisy : torch.Tensor
            Dataset with missing values added.

        mask : torch.Tensor
            Mask of missing values (1 if the value is missing, 0 otherwise).
            
        """
    noise_level = config['noise_level']
    replacement = config['replacement']

    number_of_pixels_to_mask = int(noise_level * dataset.size()[1])
    x_noisy = dataset.clone()
    x_noisy = torch.flatten(x_noisy, start_dim=1, end_dim=-1)
    mask = torch.zeros_like(x_noisy)
    for i in range(dataset.size()[0]):
        starting_pixel = torch.randint(low=0, high=int(dataset.size()[1]-number_of_pixels_to_mask), size=(1,))
        ending_pixel = starting_pixel + number_of_pixels_to_mask
        mask[i, starting_pixel:ending_pixel] = 1

    mask = mask.bool()
    if replacement == 'uniform':
        replacement = torch.randn_like(x_noisy)
        x_noisy[mask] = replacement[mask]
    else:
        x_noisy[mask] = replacement

    return x_noisy, mask.float()

def missingness_adder_fixed_patch(dataset, config):
    """
        Add missing values to a dataset in shape of a horizontal patch. The vertical placement of the patch is fixed in the middle.

        Parameters
        ----------
        dataset : torch.Tensor
            Dataset to which missing values will be added.

        config : dict
            Dictionary containing the parameters of the missingness mechanism. The following keys are expected:
            - noise_level: float
                Proportion of missing values to generate.
            - replacement: float/str
                Value to replace the missing values with. If 'uniform', missing values are replaced by uniform random values.

        Returns
        -------
        x_noisy : torch.Tensor
            Dataset with missing values added.

        mask : torch.Tensor
            Mask of missing values (1 if the value is missing, 0 otherwise).
            
        """
    noise_level = config['noise_level']
    replacement = config['replacement']

    # mask the middle noise_level proportion of the image
    starting_pixel = int(((1-noise_level)/2) * dataset.size()[1])
    ending_pixel = int(((1+noise_level)/2) * dataset.size()[1])

    x_noisy = dataset.clone()
    x_noisy = torch.flatten(x_noisy, start_dim=1, end_dim=-1)
    mask = torch.zeros_like(x_noisy)

    mask[:, starting_pixel:ending_pixel] = 1

    mask = mask.bool()
    if replacement == 'uniform':
        replacement = torch.randn_like(x_noisy)
        x_noisy[mask] = replacement[mask]
    else:
        x_noisy[mask] = replacement

    return x_noisy, mask.float()

def missingness_adder_threshold(dataset, config):
    """
        Add missing values to a dataset by covering the upper q-quantile of values pre image.

        Parameters
        ----------
        dataset : torch.Tensor
            Dataset to which missing values will be added.

        config : dict
            Dictionary containing the parameters of the missingness mechanism. The following keys are expected:
            - noise_level: float
                Proportion of missing values to generate.
            - replacement: float/str
                Value to replace the missing values with. If 'uniform', missing values are replaced by uniform random values.

        Returns
        -------
        x_noisy : torch.Tensor
            Dataset with missing values added.

        mask : torch.Tensor
            Mask of missing values (1 if the value is missing, 0 otherwise).
            
        """
    noise_level = config['noise_level']
    replacement = config['replacement']

    x_noisy = dataset.clone()
    x_noisy = torch.flatten(x_noisy, start_dim=1, end_dim=-1)

    number_vars_affected = int(noise_level * x_noisy.size(1))
    top_quantile = torch.mean(torch.topk(input=x_noisy, k=number_vars_affected, dim=1, sorted=True)[0][:,-1])
    
    mask = torch.zeros_like(x_noisy)
    mask[x_noisy > top_quantile] = 1
        
    mask = mask.bool()
    if replacement == 'uniform':
        replacement = torch.randn_like(x_noisy)
        x_noisy[mask] = replacement[mask]
    else:
        x_noisy[mask] = replacement

    return x_noisy, mask.float()

def missingness_adder_patches(dataset, config):
    """
        Add missing values to a dataset in shape of multiple randomly places patches.

        Parameters
        ----------
        dataset : torch.Tensor
            Dataset to which missing values will be added.

        config : dict
            Dictionary containing the parameters of the missingness mechanism. The following keys are expected:
            - noise_level: float
                Proportion of missing values to generate.
            - replacement: float/str
                Value to replace the missing values with. If 'uniform', missing values are replaced by uniform random values.
            - patch_size_axis: int
                Size of the patches to be placed. This value represents the length of one side of the square patch.

        Returns
        -------
        x_noisy : torch.Tensor
            Dataset with missing values added.

        mask : torch.Tensor
            Mask of missing values (1 if the value is missing, 0 otherwise).
            
        """
    noise_level = config['noise_level']
    replacement = config['replacement']
    patch_size_axis = config['patch_size_axis'] if 'patch_size_axis' in config else 4
    sizes = [28, 28]
    sizes = [32, 32] if 'CIFAR10' in config['dataset'] else sizes

    x_noisy = dataset.clone()
    number_patches = int(noise_level * x_noisy.size()[1] / patch_size_axis**2)
    if number_patches == 0:
        number_patches = 1
    x_noisy = torch.flatten(x_noisy, start_dim=1, end_dim=-1)
    x_noisy = torch.unflatten(x_noisy, dim=1, sizes=sizes)

    mask = torch.zeros_like(x_noisy)
    start_indices_x = torch.randint(low=0, high=x_noisy.size()[1]-patch_size_axis, size=(x_noisy.size()[0], number_patches))
    start_indices_y = torch.randint(low=0, high=x_noisy.size()[2]-patch_size_axis, size=(x_noisy.size()[0], number_patches))
    for i in range(x_noisy.size()[0]):
        for j in range(number_patches):
            mask[i, start_indices_x[i][j]:start_indices_x[i][j]+patch_size_axis, start_indices_y[i][j]:start_indices_y[i][j]+patch_size_axis] = 1
    mask = mask.bool()
    if replacement == 'uniform':
        replacement = torch.randn_like(x_noisy)
        x_noisy[mask] = replacement[mask]
    else:
        x_noisy[mask] = replacement
    x_noisy = torch.flatten(x_noisy, start_dim=1, end_dim=-1)
    mask = torch.flatten(mask, start_dim=1, end_dim=-1)
    return x_noisy, mask.float()


def missingness_adder_mar(dataset, config):
    """
         Add missing values to a dataset using a MAR mechanism.

        Parameters
        ----------
        dataset : torch.Tensor
            Dataset to which missing values will be added.

        config : dict
            Dictionary containing the parameters of the missingness mechanism. The following keys are expected:
            - noise_level: float
                Proportion of missing values to generate.
            - replacement: float/str
                Value to replace the missing values with. If 'uniform', missing values are replaced by uniform random values.
            - randperm_cols: bool
                If True, the columns of the dataset are randomly permuted before adding missing values.
            - device: str
                Device on which the computation is performed.
            - average_missing_rates: str/list/tuple/torch.Tensor/np.ndarray
                A list of average missing rates for each feature. Otherwise string: If 'uniform', missing rates are randomly sampled from a uniform distribution.
                If 'normal', missing rates are randomly sampled from a multivariate normal distribution. If 'normal' is used, the covariance matrix is calculated from the dataset.
                If 'normal' is used, the covariance matrix is calculated from the dataset and the following keys are used:
                - chol_eps: float
                    Epsilon value multiplied with the identity matrix and added to the covariance matrix to ensure it is positive definite.
                - sigmoid_offset: float
                    Offset value for the sigmoid function used to transform the logits to probabilities.
                - sigmoid_k: float
                    Scaling factor for the sigmoid function used to transform the logits to probabilities.

        Returns
        -------
        x_noisy : torch.Tensor
            Dataset with missing values added.

        mask : torch.Tensor
            Mask of missing values (1 if the value is missing, 0 otherwise).
            
        """
    replacement = config['replacement']
    average_missing_rates = config['average_missing_rates']
    noise_level = config['noise_level']
    randperm_cols = config['randperm_cols']
    chol_eps = config['chol_eps']
    sigmoid_offset = config['sigmoid_offset']
    sigmoid_k = config['sigmoid_k']
    device = config['device']

    generator = torch.Generator(device=device).manual_seed(42)
    N = dataset.size()[0]
    D = dataset.size()[1]
    X = dataset.to(device).clone()
    X = torch.flatten(X, start_dim=1, end_dim=-1)

    if type(average_missing_rates) in [list, tuple, torch.Tensor, np.ndarray]:
        average_missing_rate_per_feature = torch.tensor(average_missing_rates, device=device)
        if average_missing_rate_per_feature.size()[0] != D:
            raise ValueError('average_missing_rates must have the same length as the number of features')
    elif average_missing_rates == 'uniform':
        average_missing_rate_per_feature = torch.clamp(torch.rand_like(X[0], device=device)-0.5 + noise_level, 0, 1)
    elif average_missing_rates == 'normal':
        cov = torch.cov(torch.transpose(X, 0, 1))
        L = torch.linalg.cholesky(cov + chol_eps*torch.eye(cov.size()[0], device=device))
        normal_dist = torch.distributions.MultivariateNormal(torch.zeros(cov.size()[1], device=device), scale_tril=L)
        logits = normal_dist.sample()
        average_missing_rate_per_feature = torch.sigmoid((logits-sigmoid_offset)*sigmoid_k)
    else:
        raise ValueError('average_missing_rates must be either uniform, normal or a list/tuple/torch.Tensor')

    feature_weights = torch.rand_like(X[0], device=device)*2 - 1
    feature_weights = torch.diag(feature_weights)
    feature_biases = torch.rand_like(X[0], device=device)

    current_mask = torch.zeros_like(X, device=device)
    mask_probs = torch.zeros_like(X, device=device)

    # randomly change the order of the columns in dataset
    if randperm_cols:
        random_indices = torch.randperm(D)
        X = X[:, random_indices]

    A = torch.matmul(X, feature_weights)
    gamma = torch.zeros_like(X[:, None, 0], device=device)

    for feature in range(D):
        # calculate new part of summation in exponent
        if feature != 0:
            gamma += torch.diag(torch.matmul(A[:, None, feature-1], torch.transpose(1-current_mask[:, None, feature-1], 0, 1))).unsqueeze(1) + feature_biases[feature-1] * current_mask[:, None, feature-1]
            
        # calculate mask probabilities
        mask_probs[:, None, feature] = average_missing_rate_per_feature[feature] * N * torch.exp(-gamma) / torch.sum(torch.exp(-gamma))

        # replace all mask_prob values larger than 1 by 1
        mask_probs[mask_probs > 1] = 1
        if mask_probs.isnan().any():
            gamma = torch.zeros_like(X[:, None, 0], device=device)
            mask_probs[mask_probs.isnan()] = 0
            
        # sample mask
        current_mask[:, None, feature] = torch.bernoulli(mask_probs[:, None, feature], generator=generator)

    # reverse the column permutation
    if randperm_cols:
        random_indices = torch.argsort(random_indices)
        current_mask = current_mask[:, random_indices]
        X = X[:, random_indices]

    X_corrupted = X.clone()
    mask = current_mask.bool()
    if replacement == 'uniform':
        replacement = torch.randn_like(X_corrupted)
        X_corrupted[mask] = replacement[mask]
    else:
        X_corrupted[mask] = replacement
    print('proportion of masked features: ', torch.sum(mask)/torch.numel(mask))
    return X_corrupted, mask.float()

def missingness_adder_mnar(dataset, config):
    """
         Add missing values to a dataset using a MNAR mechanism.

        Parameters
        ----------
        dataset : torch.Tensor
            Dataset to which missing values will be added.

        config : dict
            Dictionary containing the parameters of the missingness mechanism. The following keys are expected:
            - noise_level: float
                Proportion of missing values to generate.
            - replacement: float/str
                Value to replace the missing values with. If 'uniform', missing values are replaced by uniform random values.
            - randperm_cols: bool
                If True, the columns of the dataset are randomly permuted before adding missing values.
            - device: str
                Device on which the computation is performed.
            - average_missing_rates: str/list/tuple/torch.Tensor/np.ndarray
                A list of average missing rates for each feature. Otherwise string: If 'uniform', missing rates are randomly sampled from a uniform distribution.
                If 'normal', missing rates are randomly sampled from a multivariate normal distribution. If 'normal' is used, the covariance matrix is calculated from the dataset.
                If 'normal' is used, the covariance matrix is calculated from the dataset and the following keys are used:
                - chol_eps: float
                    Epsilon value multiplied with the identity matrix and added to the covariance matrix to ensure it is positive definite.
                - sigmoid_offset: float
                    Offset value for the sigmoid function used to transform the logits to probabilities.
                - sigmoid_k: float
                    Scaling factor for the sigmoid function used to transform the logits to probabilities.
                    
        Returns
        -------
        x_noisy : torch.Tensor
            Dataset with missing values added.

        mask : torch.Tensor
            Mask of missing values (1 if the value is missing, 0 otherwise).
            
        """
    replacement = config['replacement']
    average_missing_rates = config['average_missing_rates']
    noise_level = config['noise_level']
    randperm_cols = config['randperm_cols']
    chol_eps = config['chol_eps']
    sigmoid_offset = config['sigmoid_offset']
    sigmoid_k = config['sigmoid_k']
    device = config['device']
    dependence = config['dependence']

    generator = torch.Generator(device=device).manual_seed(42)
    N = dataset.size()[0]
    D = dataset.size()[1]
    X = dataset.to(device).clone()
    X = torch.flatten(X, start_dim=1, end_dim=-1)

    if type(average_missing_rates) in [list, tuple, torch.Tensor, np.ndarray]:
        average_missing_rate_per_feature = torch.tensor(average_missing_rates, device=device)
        if average_missing_rate_per_feature.size()[0] != D:
            raise ValueError('average_missing_rates must have the same length as the number of features')
    elif average_missing_rates == 'uniform':
        average_missing_rate_per_feature = torch.clamp(torch.rand_like(X[0], device=device)-0.5 + noise_level, 0, 1)
    elif average_missing_rates == 'normal':
        cov = torch.cov(torch.transpose(X, 0, 1))
        L = torch.linalg.cholesky(cov + chol_eps*torch.eye(cov.size()[0], device=device))
        normal_dist = torch.distributions.MultivariateNormal(torch.zeros(cov.size()[1], device=device), scale_tril=L)
        logits = normal_dist.sample()
        average_missing_rate_per_feature = torch.sigmoid((logits-sigmoid_offset)*sigmoid_k)
    else:
        raise ValueError('average_missing_rates must be either uniform, normal or a list/tuple/torch.Tensor')

    feature_weights = torch.rand_like(X[0], device=device)*2 - 1
    feature_weights = torch.diag(feature_weights)
    feature_biases = torch.rand_like(X[0], device=device)

    current_mask = torch.ones_like(X, device=device)
    mask_probs = torch.zeros_like(X, device=device)

    # randomly change the order of the columns in dataset
    if randperm_cols:
        random_indices = torch.randperm(D)
        X = X[:, random_indices]

    A = torch.matmul(X, feature_weights)
    gamma = torch.zeros_like(X[:, None, 0], device=device)
    
    for feature in range(D):
        if dependence == 'simple_unobserved':
            # simple case (as in GAIN paper): missingness depends only on the unobserved value of the feature itself
            # gamma = feature_weights[feature][feature]*X[:, None, feature]
            gamma = A[:, None, feature]

        elif dependence == 'complex_unobserved':
            # more complex case (extends GAIN paper): missingness depends on the unobserved value of the feature itself and the unobserved values of other features
            # calculate new part of summation in exponent
            if feature != 0:
                gamma += torch.diag(torch.matmul(A[:, None, feature], torch.transpose(current_mask[:, None, feature-1], 0, 1))).unsqueeze(1)
                # here current_mask instead of 1-current_mask
                # here feature_biases removed
                # here cols of A go up to and including feature itself (that's why current_mask is initialized with ones)

        elif dependence == 'unobserved_and_observed':
            # even more complex case (extends GAIN paper): missingness depends on the unobserved value of the feature itself 
            # and the unobserved values of other features and the observed values of other features
            if feature != 0:
                gamma += (torch.diag(torch.matmul(A[:, None, feature-1], torch.transpose(1-current_mask[:, None, feature-1], 0, 1))).unsqueeze(1) + 
                          feature_biases[feature-1] * (current_mask[:, None, feature-1]) + 
                          A[:, None, feature])
        else:
            raise ValueError('dependence must be either simple_unobserved, complex_unobserved or unobserved_and_observed')

        # calculate mask probabilities
        mask_probs[:, None, feature] = average_missing_rate_per_feature[feature] * N * torch.exp(-gamma) / torch.sum(torch.exp(-gamma))

        # replace all mask_prob values larger than 1 by 1
        mask_probs[mask_probs > 1] = 1
        if mask_probs.isnan().any():
            print('had to replace nans')
            gamma = torch.zeros_like(X[:, None, 0], device=device)
            mask_probs[mask_probs.isnan()] = 0
                
        # sample mask
        current_mask[:, None, feature] = torch.bernoulli(mask_probs[:, None, feature], generator=generator)

    # reverse the column permutation
    if randperm_cols:
        random_indices = torch.argsort(random_indices)
        current_mask = current_mask[:, random_indices]
        X = X[:, random_indices]

    X_corrupted = X.clone()
    mask = current_mask.bool()
    if replacement == 'uniform':
        replacement = torch.randn_like(X_corrupted)
        X_corrupted[mask] = replacement[mask]
    else:
        X_corrupted[mask] = replacement
    return X_corrupted, mask.float()

        
def add_noise_with_model(noise_model, encoder, data, corruption_share=-1, replacement=0, additional_noise=0, generator=None, device='cuda', benchmark_noise=False, dcon=None):
        """
        Add missing values to a dataset using a noise model (like a trained MP model). This method is used to generate corruption masks for the SDAE training. 
        It is also used in the training of the Encoder and the Benchmark DAE where only uniform random noise is passed as the noise model.

        Parameters
        ----------
        noise_model : function
            Noise model used to generate the missing values. Can be a trained model or one of the missingness generating functions.
            If it is one of the missingness generating functions, benchmark_noise must be set to True, otherwise False. 
            If it is one of the missingness generating functions, dcon should be the data config dictionary and not None.

        encoder : nn.Module
            Encoder used to encode the data before passing it to the noise model. If None, the data is passed directly to the noise model.

        data : torch.Tensor
            Dataset to which missing values will be added.

        corruption_share : float, default=-1
            Proportion of variables to be corrupted. If -1, the proportion is determined by the noise model.

        replacement : float/str, default=0
            Value to replace the missing values with. If 'uniform', missing values are replaced by uniform random values.

        additional_noise : float, default=0
            Proportion of additional noise to add to the missing values.

        generator : torch.Generator, default=None
            Generator used to generate random numbers. If None, a new generator is created.

        device : str, default='cuda'
            Device on which the computation is performed.

        benchmark_noise : bool, default=False
            Needs to be set to true if the noise model is one of the missingness generating functions.
            If True, the noise model is expected to take as arguments the data and the config dictionary. 
            If True, the noise model is epxected to return a tuple of (x_noisy, mask) instead of mask_probs.

        dcon : torch.Tensor, default=None
            Data used to condition the noise model. If None, the data is not conditioned.

        Returns
        -------
        x_noisy : torch.Tensor
            Dataset with missing values added.

        mask : torch.Tensor
            Mask of missing values (1 if the value is missing, 0 otherwise).

        """
        if generator is None:
            generator = torch.Generator(device=torch.device(device)).manual_seed(42)

        x_noisy = data.clone()
        x_noisy = torch.flatten(x_noisy, start_dim=1, end_dim=-1)
        x_embedded = encoder(x_noisy.to(device)) if encoder is not None else x_noisy

        mask_probs = noise_model(x_embedded) if not benchmark_noise else noise_model(x_embedded, dcon)[1]

        if corruption_share == -1:
            mask = torch.bernoulli(mask_probs, generator=generator)
            mask = mask.bool()

            # compute additional noise
            if additional_noise > 0:
                number_vars_affected = int(additional_noise * x_noisy.size()[1])
                random_probs = torch.rand_like(mask_probs, device=device)
                top_quantile = torch.topk(input=random_probs, k=number_vars_affected, dim=1, sorted=True)[0][:,-1:]
                mask_additional = random_probs >= top_quantile
                mask = mask | mask_additional
        else:
            number_vars_affected = int(corruption_share * x_noisy.size()[1])
            if number_vars_affected == 0:
                number_vars_affected = 1
            top_quantile = torch.topk(input=mask_probs, k=number_vars_affected, dim=1, sorted=True)[0][:,-1:]
            mask = mask_probs >= top_quantile
        
        if replacement == 'uniform':
            replacement = torch.randn_like(x_noisy)
            x_noisy[mask] = replacement[mask]
        else:
            x_noisy[mask] = replacement
        return x_noisy, mask.float()



### The following code is adapted from https://github.com/BorisMuzellec/MissingDataOT: 
### In the experiments for the imputeLM model in this project, only the MNAR_mask_quantiles function is used. It is called QMNAR in the thesis.
def MAR_mask(dataset, config):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    dataset : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """
    p = config['p']
    p_obs = config['p_obs']
    replacement = config['replacement']

    x_noisy = dataset.clone()
    n, d = x_noisy.shape

    to_torch = torch.is_tensor(x_noisy) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        x_noisy = torch.from_numpy(x_noisy)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(x_noisy, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(x_noisy[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(x_noisy[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    if replacement == 'uniform':
        replacement = torch.randn_like(x_noisy)
        x_noisy[mask] = replacement[mask]
    else:
        x_noisy[mask] = replacement
    return x_noisy, mask.float()


def MNAR_mask_logistic(dataset, config):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of inputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.

    Parameters
    ----------
    dataset : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).

    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """
    p = config['p']
    p_params = config['p_params']
    exclude_inputs = config['exclude_inputs']
    replacement = config['replacement']
    
    x_noisy = dataset.clone()
    n, d = x_noisy.shape

    to_torch = torch.is_tensor(x_noisy) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        x_noisy = torch.from_numpy(x_noisy)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d ## number of variables used as inputs (at least 1)
    d_na = d - d_params if exclude_inputs else d ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    ### Other variables will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(x_noisy, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(x_noisy[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(x_noisy[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    if replacement == 'uniform':
        replacement = torch.randn_like(x_noisy)
        x_noisy[mask] = replacement[mask]
    else:
        x_noisy[mask] = replacement

    return x_noisy, mask.float()

def MNAR_self_mask_logistic(dataset, config):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """
    p = config['p']
    replacement = config['replacement']

    x_noisy = dataset.clone()
    n, d = x_noisy.shape

    to_torch = torch.is_tensor(x_noisy) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        x_noisy = torch.from_numpy(x_noisy)

    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(x_noisy, self_mask=True)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(x_noisy, coeffs, p, self_mask=True)

    ps = torch.sigmoid(x_noisy * coeffs + intercepts)

    ber = torch.rand(n, d) if to_torch else np.random.rand(n, d)
    mask = ber < ps if to_torch else ber < ps.numpy()

    if replacement == 'uniform':
        replacement = torch.randn_like(x_noisy)
        x_noisy[mask] = replacement[mask]
    else:
        x_noisy[mask] = replacement

    return x_noisy, mask.float()


def MNAR_mask_quantiles(dataset, config):
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    q : float
        Quantile level at which the cuts should occur

    p_params : float
        Proportion of variables that will have missing values

    cut : 'both', 'upper' or 'lower', default = 'both'
        Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.
        
    MCAR : bool, default = True
        If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.
        
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """
    p = config['p']
    q = config['q']
    p_params = config['p_params']
    cut = config['cut']
    MCAR = config['MCAR']
    replacement = config['replacement']

    x_noisy = dataset.clone()
    n, d = x_noisy.shape

    to_torch = torch.is_tensor(x_noisy) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        x_noisy = torch.from_numpy(x_noisy)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_na = max(int(p_params * d), 1) ## number of variables that will have NMAR values

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(d, d_na, replace=False) ### select at least one variable with missing values

    ### check if values are greater/smaller that corresponding quantiles
    if cut == 'upper':
        quants = quantile(x_noisy[:, idxs_na], 1-q, dim=0)
        m = x_noisy[:, idxs_na] >= quants
    elif cut == 'lower':
        quants = quantile(x_noisy[:, idxs_na], q, dim=0)
        m = x_noisy[:, idxs_na] <= quants
    elif cut == 'both':
        u_quants = quantile(x_noisy[:, idxs_na], 1-q, dim=0)
        l_quants = quantile(x_noisy[:, idxs_na], q, dim=0)
        m = (x_noisy[:, idxs_na] <= l_quants) | (x_noisy[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = torch.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
    ## Add a mcar mecanism on top
        mask = mask | (torch.rand(n, d) < p)

    if replacement == 'uniform':
        replacement = torch.randn_like(x_noisy)
        x_noisy[mask] = replacement[mask]
    else:
        x_noisy[mask] = replacement

    return x_noisy, mask.float()

# internal helper
def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs

# internal helper
def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts

def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.

    q : float
        Quantile level (starting from lower values).

    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.


    Returns
    -------
        quantiles : torch.DoubleTensor

    """
    return X.kthvalue(int(q * X.size(dim)), dim=dim)[0]



