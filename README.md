# CrystalCGAIN: A Generative Adversarial Imputation Network for Predicting Porous Crystal Structures with Targeted Property
This is a PyTorch implementation of CrystalCGAIN model and discussion experiments proposed by our paper "CrystalCGAIN: A Generative Adversarial Imputation Network for Predicting Porous Crystal Structures with Targeted Property".
# 1. Overview
![Graphic Abstract](https://github.com/user-attachments/assets/ef4ee919-d5f9-400e-b681-e2ca8ff52021)
Fig.1 An overview of CrystalCGAIN model.
This model employs a concise inversion-free representation method and uses a Generative Adversarial Imputation Network (GAIN) to generate new crystal structures. The crystal structures are represented by lattice parameters and fractional atomic coordinates, combined with atomic distance matrices as inputs to the model. By utilizing random missing sampling, the data distribution of crystal structures is explored. Additionally, a soft constraint loss function for target property-oriented generation is designed, enabling the generation of new crystal structures that meet user-defined property expectations.
# 2. Installation
Set up a python environment for version 3.9.4 and clone the Github repo.
## 2.1 Installed Python Packages
| Packge               | Version       | Packge                | Version        | Packge                 | Version        |
|---------------------|-------------|---------------------|-------------|---------------------|-------------|
| ase                 | 3.22.1     | matplotlib          | 3.7.4      | pymatgen            | 2023.8.10  |
| torch               | 1.13.0+cu116 | torchaudio          | 0.13.0+cu116 | torchvision         | 0.14.0+cu116 |

## 2.2 Software Installation and Path Configuration
Before running the code, you need to install the **Zeo++** software by following the installation steps provided on its official website: [Zeo++ Official Website](https://www.zeoplusplus.org/about.html).

### Path Configuration
After installing the software:
1. Modify the absolute paths in the provided code to match your local setup. These paths include:
   - The path to the dataset.
   - The path to the **Zeo++** software.

2. Refer to the examples on the Zeo++ website for details on running the software with sample inputs.

By correctly configuring the paths, you ensure seamless integration of **Zeo++** with the project code.

# 3. Datasets
You can download the datasets via reference URL in the follow table.
| Datasets  | Description  | Reference  |
|---------|--------------------------------------------------------|----------------------------------------------|
|PCOD  |The goal of this repository is to serve as stable hosting (mirror) for some existing databases of zeolitic structures.  |https://github.com/fxcoudert/zeolite_databases  |
|IZA  |This database provides structural information on all of the Zeolite Framework Types that have been approved by the Structure Commission of the International Zeolite Association (IZA-SC).  |  [www.iza-structure.org/databases](https://www.iza-structure.org/databases/)|

# 4. Training and Generation
To train the CrystalCGAIN, you have to exec the following commands.
## 4.1 Train Reconstruction
Description: This section of the files is used to test the reconstruction of existing crystal structures.
1. Extract the origin_data archive.
2. Run the main_runner.py file.
3. The code will automatically execute each part. Once training is complete, it will save the training process and output an image showing the RMSE changes during training.
4. Open the example folder to check if the output matches the provided example.
## 4.2 Train Generation
Description: This section is used to test the model's generation and training process.
1. Extract the origin_data archive.
2. Run the main_runner.py file.
3.The code will automatically run each part and save the training results in two separate folders: gan_model_step_1 and gan_model_step_2.
4. Once the code has fully run, an image will be output showing the dramatic decrease in void fraction as the training epochs progress.
5. By opening the example folder, you will find the saved model parameters from both the pre-trained model and the model after incorporating the target property-directed generation module, along with the training process logs.
## 4.3 Generate Structure
Description: This section is used to test the generation of new structures.
1. Extract the origin_data archive.
2. Run gen_fake_batch_cif_input_output.py file.
   - Open the "fake_data" folder in the "example" directory, unzip the two compressed datasets, and use them for testing input-output comparison.
   - These two files occupy a large amount of storage space and cannot be uploaded to GitHub. If necessary, you can contact the author for assistance.(Contact email: caize@shu.edu.cn)
3. Run gen_fake_calculate_void_fraction.py file.
   - It will output a porosity distribution graph, which can be compared with the example graph in the "example" folder to see if they have a similar distribution trend.
## 4.4 New Structure
Opening this folder will provide information on the new crystal structures:

   - CIF files of 33 stable structures and 83 metastable structures selected by M3GNet.
   - Excel sheets containing the calculated parameters related to voids.
   - Data statistics images.
   - Zeo++ example run code (requires modification of absolute paths).

# 5. Acknowledgements
This work was sponsored by the Key Program of Science and Technology of Yunnan Province (No.202302AB080020, 202102AB080019-3), Key Research Project of Zhejiang Laboratory (No. 2021PE0AC02), Key Project of Shanghai Zhangjiang National Independent Innovation Demonstration Zone(No. ZJ2021-ZD-006). The authors gratefully appreciate the anonymous reviewers for their valuable comments.
