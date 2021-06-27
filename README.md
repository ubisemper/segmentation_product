# segmentation_product
This repository contains the scripts for reproducing the research we have done on hair segmentation.
The software is build upon the code from [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## General infomation
- The required packages can be found in the requirements.txt file.

- The models folder, is the folder where the trained models will be stored. Also the pretrained models from earlier research can be found there, for further evaluation.

- The Data folder contains the Figaro1K dataset, and the Celeb1k dataset.
The full preprocessed haisegmentation dataset from CelebA can be found here: [CelebA mask processed](https://drive.google.com/file/d/16WEPwGDfCgLoi6t_1-VxzJqEyUSleNL8/view?usp=sharing)

## seg_train.py
This is the main script to train several models finetuned for hair segmentation

- Commands 
  - `--epoch` Number of epoch to train the model, default is 40
  - `--plot` Choose to save the loss and IOU score figures
  - `--model_name` Name of the model that is going to be saved
  - `--data_type` Switch between figaro and celebA dataset. Because of the different indexing names the dataset uses
  - `--model` Choose which model to use for training, Unet, FPN, PAN, DeepLabV3, UnetPlusPlus
  - `--bach_size` Specify training batch size.
