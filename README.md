# Anomaly Detection Framework Using Multiple Techniques

## Introduction
This project explores multiple anomaly detection techniques applied to multivariate time series data, extending beyond contrastive learning and Generative Adversarial Networks (GAN) to include Principal Component Analysis (PCA), Local Outlier Factor (LOF), Dynamic Time Warping (DTW), and Isolation Forest. This diverse toolkit allows for robust anomaly detection across various scenarios, enhancing the framework's ability to handle different data distributions and anomaly types.

## Dependencies
Python 3.8+
Pandas
TensorFlow 2.x
Scikit-Learn
Numpy
Seaborn
Installation

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data.git
cd Anomaly-Detection-in-Multivariate-Time-Series-Data
```

## Dataset
This project uses the PSM dataset provided by eBay. The dataset can be accessed at this GitHub repository TS-AD-Datasets.

![image](https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data/assets/76420523/7f780968-7b5a-4edb-a3e9-91f0d6ece03f)

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
Data loading and preprocessing using Pandas.
Data Visualization using seaborn.
Data scaling and transformation using Scikit-Learn's StandardScaler.
Data augmentation using geometric distribution masks.
Implementation of various anomaly detection models:
Autoencoder
PCA
LOF
DTW
Isolation Forest

## Implementation Details
#### Data Loading and Preprocessing
The framework begins by loading the dataset and preprocessing it using pandas DataFrames. This step ensures that the data is ready for further processing and model training.

#### Data Visualization 
For initial data exploration, we utilized Seaborn to create pair plots of the training data. This step was crucial for understanding the relationships and distributions of different features within our data, helping us pinpoint potential areas where anomalies might be more apparent

![image](https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data/assets/76420523/7f0afdd4-7f29-4635-9d50-3b55583afc5a)

#### Data Preparation 
Data preparation involved cleaning and scaling to ensure the quality and consistency of our analysis. We replaced infinite values with NaN and dropped all missing values. Subsequently, we applied MinMaxScaler for normalization to standardize the range of feature values, which is particularly important for the performance of many anomaly detection algorithms. 

![image](https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data/assets/76420523/9e3e19a5-3b18-469a-88b7-7744dc5b98bf)


#### Geometric Masking
Geometric distribution masks are applied to the data as a data augmentation technique. This helps in introducing variability into the training data while preserving the underlying patterns.


![image](https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data/assets/76420523/0f785228-14b5-47f3-b2af-235edd40cf59)


## Anomaly Detection Techniques 
#### Autoencoder
Employs geometric masking for data augmentation and contrastive loss for training, focusing on reconstruction accuracy to detect anomalies.

##### Model Training 
I constructed an autoencoder with a contrastive loss function to learn and reproduce normal patterns within the data. The training process involved careful tuning of network parameters to balance between overfitting and underfitting. 

![image](https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data/assets/76420523/d20728de-640e-46d8-b81a-f482bea70b9f)

Using the trained autoencoder, we calculated anomaly scores based on the reconstruction errors. 

![image](https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data/assets/76420523/db5bae2e-579e-4d19-a20a-0597443acfde)

#### PCA
Reduces dimensionality to highlight the most significant features, using reconstruction errors to identify anomalies.

![image](https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data/assets/76420523/8cd3d6e5-4526-435d-b382-dfac56c0eb54)

#### LOF
Identifies anomalies by measuring the local deviation of a data point with respect to its neighbors.

![image](https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data/assets/76420523/dd5718c8-65ce-47bb-a515-87959fbd2380)

#### DTW
I employed DTW to compare time-series data, allowing us to identify anomalies based on the similarity of sequences through dynamic alignment. We apply DTW to the training and test data, comparing each test sample to the nearest training sample. 

![image](https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data/assets/76420523/c197b7a9-8440-49ef-81b2-852a05e54226)

#### Isolation Forest
I also explored the use of Isolation Forest, which isolates anomalies instead of profiling normal data points. This method is particularly effective for handling high-dimensional data. 
![image](https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data/assets/76420523/61e6437d-d5df-471a-9a4c-d3b8440e3cb0)

### Empirical Analysis of methods. 
The trained Autoencoder is evaluated on a separate test dataset to detect anomalies. Reconstruction errors are computed for each data sample, and anomalies are identified based on predefined thresholds.

## Results
The performance of each technique was visualized and documented, showing their respective precision, recall, F1-scores, and AUC values. Based on our findings, we concluded that the choice of anomaly detection method heavily depends on the nature of the dataset and the specific types of anomalies one expects to encounter. 

![image](https://github.com/Maryam189/Anomaly-Detection-in-Multivariate-Time-Series-Data/assets/76420523/e4044f52-e9b9-401e-803a-3708744923ec)


## Contributing
Contributions to this project are welcome. Please submit a pull request with your features and bug fixes.

## Useful Links
For your convenience, you can access public datasets for time series anomaly detection, such as the PSM dataset from eBay, using the following link: [TS-AD-Dataset](https://github.com/elisejiuqizhang/TS-AD-Datasets)
