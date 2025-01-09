import argparse
import numpy as np
import torch
from crystal_data_loader import data_loader, train_test_split_data, data_distance_matrix_loader
from crystal_utils import rmse_loss, rmse_obse_loss
from model_reconstruction import reconstruction


def main(args, model):
    data_name = args.data_name
    adj_name = args.adj_name
    miss_number = args.miss_number

    gain_parameters = {
        'batch_size': args.batch_size,
        'hint_rate': args.hint_rate,
        'alpha': args.alpha,
        'epochs': args.epochs
    }

    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_number)
    adj_data = data_distance_matrix_loader(adj_name)
    train_x, test_x, train_miss_x, test_miss_x, train_m, test_m, train_adj, test_adj \
        = train_test_split_data(ori_data_x, miss_data_x, data_m, adj_data)

    # Impute missing data using PyTorch GAIN function
    train_rmse, test_rmse = model(train_x, test_x, train_miss_x, test_miss_x, train_m, test_m, train_adj, test_adj, gain_parameters)

    return train_rmse, test_rmse


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
        default=2,
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
        default=100,
        type=float
    )
    parser.add_argument(
        '--epochs',
        help='number of training epochs',
        default=3000,
        type=int
    )

    args = parser.parse_args()
    model = reconstruction

    # Calls main function
    train_rmse, test_rmse = main(args, model)
