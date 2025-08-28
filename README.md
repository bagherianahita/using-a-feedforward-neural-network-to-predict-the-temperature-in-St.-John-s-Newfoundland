# Temperature Prediction with a Feedforward Neural Network

This project implements a neural network model to predict the temperature for the next six hours in St. John's, Newfoundland, based on historical weather data. The notebook explores and compares the performance of a simple baseline model against a custom-built feedforward neural network (NN).

## Key Features

- **Data Preprocessing**: T  `MinMaxScaler` to normalize the temperature data to a range of 0 to 1.
- **Data Splitting**: The dataset is split into 80% for training and 20% for testing.
- **Model Development**: A feedforward neural network with multiple hidden layers and a variable number of units is developed and trained.
- **Baseline Models**: A simple baseline model is created, which assumes the temperature will remain the same as the last observed temperature.
- **Performance Evaluation**: The models are verified on the test dataset using the Mean Squared Error (MSE) metric.

## Requirements

- `netCDF4`
- `numpy`
- `cftime`
- `tensorflow`
- `keras`
