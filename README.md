# Anomaly Detection Framework Using Contrastive Learning and GAN

This project implements an anomaly detection framework that integrates contrastive learning and Generative Adversarial Networks (GANs) to mitigate overfitting in multivariate time series data. The framework utilizes a Transformer-based Autoencoder and data augmentation techniques employing geometric distribution masks.

## Getting Started

### Dependencies

- Python 3.8+
- Pandas
- TensorFlow 2.x
- Scikit-Learn
- Numpy

### Installation

Clone the repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

Dataset
This project uses the PSM dataset provided by eBay. The dataset can be accessed at this GitHub repository.

Files
train.csv: Training data file.
test.csv: Test data file.
test_label.csv: Test labels indicating anomalies.
Usage
Run the script to start the training and evaluation process:

```bash
python <script-name>.py

Project Structure
The code is organized as follows:

Data loading and preprocessing using Pandas.
Data scaling and transformation using Scikit-Learn's StandardScaler.
Data augmentation using a geometric distribution mask.
Building and training a Transformer-based Autoencoder with TensorFlow.
Building and training a discriminator model as part of the GAN framework.
Evaluating the model on the test set and calculating reconstruction errors.
Models
Transformer-based Autoencoder
The Autoencoder uses multi-head attention to capture temporal correlations in time series data. The encoder-decoder architecture facilitates the learning of compressed representations.

Discriminator
The discriminator assesses the realism of the reconstructed data, playing a crucial role in the GAN training loop.

Anomaly Detection
Anomalies are detected based on thresholds computed from the reconstruction errors and realism scores. The framework identifies data points that deviate significantly from the model's learned distribution.

Results
Results include the reconstruction error statistics and realism scores. Detected anomalies are summarized and presented.

Contributing
Contributions to this project are welcome. Please fork the repository and submit pull requests with your features and bug fixes.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.
