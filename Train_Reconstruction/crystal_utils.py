import torch
import numpy as np

def normalization(data, parameters=None):
    '''Normalize data in [0, 1] range.

    Args:
      - data: original data (torch tensor)

    Returns:
      - norm_data: normalized data (torch tensor)
      - norm_parameters: min_val, max_val for each feature for renormalization
    '''
    data = data.numpy()  # Convert to numpy for processing
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:
        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val, 'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return torch.tensor(norm_data, dtype=torch.float32), norm_parameters

def renormalization(norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.

    Args:
      - norm_data: normalized data (torch tensor)
      - norm_parameters: min_val, max_val for each feature for renormalization

    Returns:
      - renorm_data: renormalized original data (torch tensor)
    '''
    norm_data = norm_data.numpy()  # Convert to numpy for processing
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]

    return torch.tensor(renorm_data, dtype=torch.float32)

def rounding(imputed_data, data_x):
    '''Round imputed data for categorical variables.

    Args:
      - imputed_data: imputed data (torch tensor)
      - data_x: original data with missing values (torch tensor)

    Returns:
      - rounded_data: rounded imputed data (torch tensor)
    '''
    imputed_data = imputed_data.numpy()  # Convert to numpy for processing
    data_x = data_x.numpy()  # Convert to numpy for processing
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return torch.tensor(rounded_data, dtype=torch.float32)

def rmse_loss(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data

    Args:
      - ori_data: original data without missing values (torch tensor)
      - imputed_data: imputed data (torch tensor)
      - data_m: indicator matrix for missingness (torch tensor)

    Returns:
      - rmse: Root Mean Squared Error
    '''

    # Only for missing values
    ori_data = ori_data.numpy()
    imputed_data = imputed_data.cpu().numpy()
    data_m = data_m.cpu().numpy()

    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)

    rmse = np.sqrt(nominator / float(denominator))

    return rmse

def rmse_obse_loss(ori_data, imputed_data, data_m):
    # Only for missing values
    ori_data = ori_data.numpy()
    imputed_data = imputed_data.cpu().numpy()
    data_m = data_m.cpu().numpy()

    nominator = np.sum((data_m * ori_data - (data_m * imputed_data)) ** 2)
    denominator = np.sum(data_m)

    rmse = np.sqrt(nominator / float(denominator))

    return rmse

def rmse_loss_gpu(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data

    Args:
      - ori_data: original data without missing values (torch tensor)
      - imputed_data: imputed data (torch tensor)
      - data_m: indicator matrix for missingness (torch tensor)

    Returns:
      - rmse: Root Mean Squared Error
    '''
    # Only for missing values
    nominator = torch.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = torch.sum(1 - data_m)

    rmse = torch.sqrt(nominator / float(denominator))

    return rmse.item()  # Return as a standard Python float

def rmse_loss_all(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data

    Args:
      - ori_data: original data without missing values (torch tensor)
      - imputed_data: imputed data (torch tensor)
      - data_m: indicator matrix for missingness (torch tensor)

    Returns:
      - rmse: Root Mean Squared Error
    '''

    # Only for missing values
    ori_data = ori_data.numpy()
    imputed_data = imputed_data.numpy()
    data_m = data_m.numpy()

    nominator = np.sum((ori_data - imputed_data) ** 2)
    denominator = np.sum(1 - data_m) + np.sum(data_m)

    rmse = np.sqrt(nominator / float(denominator))

    return rmse

def rmse_loss_not_miss(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data

    Args:
      - ori_data: original data without missing values (torch tensor)
      - imputed_data: imputed data (torch tensor)
      - data_m: indicator matrix for missingness (torch tensor)

    Returns:
      - rmse: Root Mean Squared Error
    '''

    # Only for missing values
    ori_data = ori_data.numpy()
    imputed_data = imputed_data.numpy()
    data_m = data_m.numpy()

    nominator = np.sum((data_m * ori_data - data_m * imputed_data) ** 2)
    denominator = np.sum(data_m)

    rmse = np.sqrt(nominator / float(denominator))

    return rmse

def xavier_init(size):
    '''Xavier initialization (also known as Glorot initialization).

    Args:
      - size: vector size

    Returns:
      - initialized random vector (torch tensor).
    '''
    in_dim = size[0]
    xavier_stddev = 1. / torch.sqrt(torch.tensor(in_dim / 2., dtype=torch.float32))
    return torch.randn(size) * xavier_stddev

def binary_sampler(p, rows, cols):
    '''Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    '''
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix

def binary_sampler_coords(m, row, col, start = 3, end = 26):
    '''Sample binary mask where exactly m coordinates are missing per sample.

    Args:
      - m: the number of coordinates to be missing
      - row: the number of coordinates (each with col dimensions)
      - col: the dimension of each coordinate (should be 3 for 3D coordinates)

    Returns:
      - binary_random_matrix: generated binary mask with m missing coordinates.
    '''
    # Create a binary mask with all ones
    binary_random_matrix = np.ones((row, col))

    # Randomly select m coordinates to be set as missing (0)
    available_indices = np.arange(start, end)
    missing_indices = np.random.choice(available_indices, m, replace=False)

    binary_random_matrix[missing_indices, :] = 0

    return binary_random_matrix

def uniform_sampler(low, high, rows, cols):
    '''Sample uniform random variables.

    Args:
      - low: low limit
      - high: high limit
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - uniform_random_matrix: generated uniform random matrix.
    '''
    return np.random.uniform(low, high, size=[rows, cols])

def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.

    Args:
      - total: total number of samples
      - batch_size: batch size

    Returns:
      - batch_idx: batch index
    '''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx
