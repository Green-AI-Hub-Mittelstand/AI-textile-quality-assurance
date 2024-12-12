# AutoEncoder for Anomaly Detection

This repository implements an AutoEncoder-based model for anomaly detection in images. The steps below describe how to train the model, test it on different anomaly types, and test it on a specific folder. 

## Requirements

Make sure you have the following dependencies installed:

```bash
pip install torch torchvision matplotlib numpy
```

## 1. Training the AutoEncoder

To train the AutoEncoder model, run the following command:

```bash
python train.py --epochs 40 --lr 0.001 --batch_size 16
```
--epochs: Number of training epochs (default: 40)
--lr: Learning rate (default: 0.001)
--batch_size: Batch size for training (default: 16)

The training process will:

Save a plot of the training loss after each epoch in ./results/loss_plot.png.
Save the model checkpoint after training completes in the form of autoencoder_{epoch}_{loss}.pth.

## 2. Testing Anomalies on New Categories
You can test the trained AutoEncoder model on new anomaly categories by running the following command:

```bash
python test.py --test_data_root Data/test2 --model_path autoencoder_320.00160.pth
```
--test_data_root: Root folder containing sub-folders of different anomaly categories. Each sub-folder should contain test images.
--model_path: Path to the trained AutoEncoder model (.pth file).
This will:

Calculate the reconstruction error for each image and plot it as a scatter plot, with each anomaly category represented by a different color.
Save the reconstruction error plot in ./results/reconstruction_error_plot.png.

## 3. Testing Anomalies on a Specific Folder
To test anomalies on a specific folder and save the results in a text file, run the following command:

```bash
python test_new.py --test_data_root Data/test2/err_material_nok --model_path autoencoder_320.00160.pth
```
--test_data_root: Folder containing images to test for anomalies.
--model_path: Path to the trained AutoEncoder model (.pth file).
The results will be saved as a text file (./results/reconstruction_errors.txt) with the following format:

```makefile
image1.jpg: reconstruction_error1
image2.jpg: reconstruction_error2
```
This file contains the reconstruction error for each image in the folder.

## Results
All results including loss plots, reconstruction error plots, and the saved model will be stored in the ./results/ directory.


