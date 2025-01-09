import argparse
import numpy as np
import torch
from crystal_data_loader import data_loader, train_test_split_data, data_distance_matrix_loader
from crystal_utils import rmse_loss, rmse_obse_loss
from gan_model_step_1 import gan_step_1

def main(args, model):
    data_name = args.data_name
    adj_name = args.adj_name
    miss_number = args.miss_number

    gain_parameters = {
        'batch_size': args.batch_size,
        'hint_rate': args.hint_rate,
        'alpha': args.alpha,
        'beta': args.beta,
        'epochs': args.epochs
    }

    # Impute missing data using PyTorch GAIN function
    train_rmse = model(gain_parameters, args)

    return train_rmse


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--adj_name',
        choices=['origin_data\PCOD_npy_sort_folder_distance_matrix_npy'],
        default='origin_data\PCOD_npy_sort_folder_distance_matrix_npy',
        type=str
    )
    parser.add_argument(
        '--data_name',
        choices=['origin_data/PCOD_npy_sort_folder_Concatenated_npy'],
        default='origin_data/PCOD_npy_sort_folder_Concatenated_npy',
        type=str
    )
    parser.add_argument(
        '--miss_number',
        help='missing data probability',
        default=8,
        type=float
    )
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int
    )
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float
    )
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=1,
        type=float
    )

    parser.add_argument(
        '--beta',
        help='hyperparameter',
        default=100,
        type=float
    )

    parser.add_argument(
        '--epochs',
        help='number of training epochs',
        default=5000,
        type=int
    )

    args = parser.parse_args()
    model = gan_step_1

    # Calls main function
    train_rmse = main(args, model)

