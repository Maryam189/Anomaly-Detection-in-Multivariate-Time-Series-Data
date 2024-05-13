# Anomaly Detection Framework Using Contrastive Learning and GAN

## Introduction
This project implements an anomaly detection framework that leverages contrastive learning and Generative Adversarial Networks (GAN) to address overfitting in multivariate time series data. The implementation integrates various techniques, including data augmentation with geometric distribution masks, a Transformer-based Autoencoder, and contrastive loss, to achieve robust anomaly detection.

## Dependencies

- Python 3.8+
- Pandas
- TensorFlow 2.x
- Scikit-Learn
- Numpy

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data.git
cd Anomaly-Detection-in-Multivariate-Time-Series-Data
```

## Dataset
This project uses the PSM dataset provided by eBay. The dataset can be accessed at this GitHub repository TS-AD-Datasets.

## Files
train.csv: Training data file.
test.csv: Test data file.
test_label.csv: Test labels indicating anomalies.
Usage
Run the script to start the training and evaluation process:

```bash
python model.py
```

## Project Structure
The code is organized as follows:

Data loading and preprocessing using Pandas.
Data scaling and transformation using Scikit-Learn's StandardScaler.
Data augmentation using a geometric distribution mask.
Building and training a Transformer-based Autoencoder with TensorFlow.
Building and training a discriminator model as part of the GAN framework.
Evaluating the model on the test set and calculating reconstruction errors.

## Implementation Details
#### Data Loading and Preprocessing
The framework begins by loading the dataset and preprocessing it using pandas DataFrames. This step ensures that the data is ready for further processing and model training.

#### Geometric Masking
Geometric distribution masks are applied to the data as a data augmentation technique. This helps in introducing variability into the training data while preserving the underlying patterns.

#### Transformer-based Autoencoder
A Transformer-based Autoencoder architecture is employed for feature extraction and reconstruction. This architecture consists of multi-head self-attention layers and feed-forward networks, which are effective in capturing temporal dependencies in time series data.

#### Training
The Autoencoder model is trained using the preprocessed data. The training process involves minimizing the mean squared error between the input and reconstructed data. This step enables the model to learn meaningful representations of the input data.

![image](https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data/assets/76420523/0f785228-14b5-47f3-b2af-235edd40cf59)


#### Discriminator
A discriminator model is built to distinguish between real and reconstructed data samples. This model is trained alongside the Autoencoder as part of the GAN framework.

#### GAN Model
The GAN model combines the Autoencoder and discriminator into a unified framework. It utilizes adversarial training to enhance the quality of the reconstructed data while simultaneously training the discriminator to distinguish between real and fake samples.

#### Evaluation
The trained Autoencoder is evaluated on a separate test dataset to detect anomalies. Reconstruction errors are computed for each data sample, and anomalies are identified based on predefined thresholds.

## Results
Results include the reconstruction error statistics and realism scores. Detected anomalies are summarized and presented.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit pull requests with your features and bug fixes.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
