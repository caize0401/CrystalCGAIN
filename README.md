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
| ase                 | 3.22.1     | blinker             | 1.8.2      | BTrees              | 5.1        |
| certifi             | 2023.11.17 | cffi                | 1.16.0     | charset-normalizer  | 3.3.2      |
| click               | 8.1.7      | colorama            | 0.4.6      | contourpy           | 1.1.1      |
| cycler              | 0.12.1     | emmet-core          | 0.68.0     | Flask               | 3.0.3      |
| fonttools           | 4.46.0     | future              | 0.18.3     | idna                | 3.6        |
| importlib_metadata  | 8.5.0      | importlib-resources | 6.1.1      | itsdangerous        | 2.2.0      |
| Jinja2              | 3.1.4      | joblib              | 1.3.2      | kiwisolver          | 1.4.5      |
| latexcodec          | 2.0.1      | MarkupSafe          | 2.1.5      | matplotlib          | 3.7.4      |
| monty               | 2023.9.25  | mp-api              | 0.35.1     | mpmath              | 1.3.0      |
| msgpack             | 1.0.7      | networkx            | 3.1        | numpy               | 1.24.4     |
| packaging           | 23.2       | palettable          | 3.3.3      | pandas              | 2.0.3      |
| patsy               | 0.5.6      | persistent          | 5.1        | Pillow              | 10.1.0     |
| pip                 | 23.3.1     | plotly              | 5.18.0     | pybtex              | 0.24.0     |
| pycparser           | 2.21       | pydantic            | 1.10.13    | pymatgen            | 2023.8.10  |
| pyparsing           | 3.1.1      | python-dateutil     | 2.8.2      | pytz                | 2023.3.post1 |
| PyYAML              | 6.0.1      | requests            | 2.31.0     | ruamel.yaml         | 0.18.5     |
| ruamel.yaml.clib    | 0.2.8      | scikit-learn        | 1.3.2      | scipy               | 1.10.1     |
| seaborn             | 0.13.2     | setuptools          | 49.2.1     | six                 | 1.16.0     |
| spglib              | 2.2.0      | statsmodels         | 0.14.1     | sympy               | 1.12       |
| tabulate            | 0.9.0      | tenacity            | 8.2.3      | threadpoolctl       | 3.2.0      |
| torch               | 1.13.0+cu116 | torchaudio          | 0.13.0+cu116 | torchvision         | 0.14.0+cu116 |
| tqdm                | 4.66.1     | transaction         | 4.0        | typing_extensions   | 4.8.0      |
| tzdata              | 2023.3     | uncertainties       | 3.1.7      | urllib3             | 2.1.0      |
| utils               | 1.0.2      | Werkzeug            | 3.0.4      | zc.lockfile         | 3.0.post1  |
| ZConfig             | 4.0        | zdaemon             | 5.0        | ZEO                 | 6.0.0      |
| zipp                | 3.20.2     | ZODB                | 5.8.1      | zodbpickle          | 3.1        |
| zope.interface      | 6.1        |                     |             |                     |             |
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
1.Extract the origin_data archive.
2.Run the main_runner.py file.
3.The code will automatically execute each part. Once training is complete, it will save the training process and output an image showing the RMSE changes during training.
4.Open the example folder to check if the output matches the provided example.
## 4.2 Train Generation

## 4.3 Generate Structure

## 4.4 New Structure
Opening this folder will provide information on the new crystal structures:

   - CIF files of 33 stable structures and 83 metastable structures selected by M3GNet.
   - Excel sheets containing the calculated parameters related to voids.
   - Data statistics images.
   - Zeo++ example run code (requires modification of absolute paths).

# 5. Acknowledgements
This work was sponsored by the Key Program of Science and Technology of Yunnan Province (No.202302AB080020, 202102AB080019-3), Key Research Project of Zhejiang Laboratory (No. 2021PE0AC02), Key Project of Shanghai Zhangjiang National Independent Innovation Demonstration Zone(No. ZJ2021-ZD-006). The authors gratefully appreciate the anonymous reviewers for their valuable comments.
