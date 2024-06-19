# VITS 
## _Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech_

> [!IMPORTANT]
> This repository has been updated to support a Portuguese(pt-br) dataset and includes various code improvements. For more details about VITS, visit the [VITS GitHub repository.](https://link-url-here.org).

## Pre-requisites
0. Python >= 3.8
0. Clone this repository 
     ```sh
    git clone https://github.com/CPqD/resid2023-fala-4.git
    ```
0. Access the directory of the forked VITS code
    ```sh
    cd vits
    ```
0. Install python requirements. Check the [requirements.txt](requirements.txt) file for the complete list of requirements
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract your dataset inside the "data" direcotory
    1. For mult-speaker setting, follow the original read.me of [VITS GitHub repository.](https://link-url-here.org) 
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
    ```sh
    # Cython-version Monotonoic Alignment Search
    cd monotonic_align
    python setup.py build_ext --inplace
    ```

## Data preparing
> [!TIP]
> This guide explains how to prepare your custom audio dataset using the [prepare_data.py](prepare_data.py). If you prefer working with a notebook file, you can follow the same steps in [prepare_data.ipynb](prepare_data.ipynb)

0. Access the Script [prepare_data.py](prepare_data.py)
0. Set the Custom Dataset Path:
    1.  Locate the ``` path_custom_dataset ``` variable in the script.
    2.  Update it with the directory path to your custom dataset (e.g., 'data/custom_dataset').
0. Adjust File Reading:
    1. Look for the line that reads the CSV file containing audio paths and sentences:
        ```sh
        df_dataset_tts_portuguese = pd.read_csv(f'{path_custom_dataset}/texts.csv', header=None, sep='==')
        ```
    2. Customize this line according to your dataset format. For example, if your input file contains “path/wav/file : Input sentence,” adjust the separator (sep) accordingly (e.g., sep=':').
0. Execute the script 
    1. Run the following command in your terminal 
        ```sh
        python prepare_data.py
        ```
    2. The script will perform the following tasks:
        1. Check if audio files exist at the specified paths and mark any missing files.
        2. Remove rows with missing sentences.
        3. Calculate the duration of each audio file using librosa.get_duration().
        4. Split the dataset into training and validation sets (80% training, 20% validation)
        5. Save the resulting train and validation CSV files (train.csv and valid.csv) in the path_custom_dataset directory.

## Data preprocessing
> [!TIP]
> The [preprocess.py](preprocess.py) script utilizes the train.csv and valid.csv files generated during the data preparation process. It is designed to work with the portuguese_cleaners configuration. If you need to modify these settings, follow the steps below.

0. Access the Script [preprocess.py](preprocess.py)
0. Adjust the script:
    1. Look for the line that define this settings:
        ```sh
        # Train and valid files created with prepare_data.py
        parser.add_argument("--filelists", nargs="+", default=["data/custom_dataset/train.csv", "data/custom_dataset/valid.csv"]) 
        # Portuguese_cleaners added at cleaners.py
        parser.add_argument("--text_cleaners", nargs="+", default=["portuguese_cleaners"]) 
        ```
    2. Customize this line according to your dataset format. For example, if your input file contains “path/wav/file : Input sentence,” adjust the separator (sep) accordingly (e.g., sep=':'). 
0. Execute the script 
    1. Run the following command in your terminal 
        ```sh
        python preprocess.py
        ```
    
## Training Exmaple
1. Run the following command in your terminal 
    ```sh
    python train.py
    ```

## Inference Example
See [inference.ipynb](inference.ipynb)
