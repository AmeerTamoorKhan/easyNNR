# easyNNR: Easy Neural Networks in R

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![R Version](https://img.shields.io/badge/R-%E2%89%A5%204.0.0-blue)](https://www.r-project.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.15.0-red)](https://keras.io/)

> **Simple, automatic, and powerful neural networks for beginners and researchers**

easyNNR is a comprehensive R package that allows you to build, train, and evaluate neural networks with minimal code while offering advanced capabilities for experienced users. Whether you're a student learning machine learning or a researcher needing quick prototypes, easyNNR handles the complexity for you.

---

## 📚 Table of Contents

- [Why easyNNR?](#-why-easynr)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Complete Function Reference](#-complete-function-reference)
- [Training Parameters Guide](#-training-parameters-guide)
- [Preprocessing Options](#-preprocessing-options)
- [Advanced Examples](#-advanced-examples)
- [Model Persistence](#-model-persistence)
- [Visualization](#-visualization)
- [Prediction Functions](#-prediction-functions)
- [Best Practices](#-best-practices)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🌟 Why easyNNR?

### Key Features

✅ **One-Function Solution** - Train complex neural networks with a single `easy_nn()` call  
✅ **Automatic Preprocessing** - 50+ preprocessing techniques handled automatically  
✅ **Smart Defaults** - Works out-of-the-box with sensible defaults  
✅ **Advanced Capabilities** - Full control over architecture, optimization, and preprocessing  
✅ **Production Ready** - Complete model save/load system for deployment  
✅ **Beautiful Visualizations** - Publication-quality plots with ggplot2  
✅ **Comprehensive Documentation** - Every function thoroughly documented  
✅ **Reproducible** - Built-in seed management and logging  

### What Makes It Different?

Traditional neural network workflow in R:
```r
# 😫 Traditional approach: ~100+ lines of code
library(keras)
# Data preprocessing
data_scaled <- scale(data[, -target_col])
# One-hot encoding
# Feature engineering
# Train/test split
# Build model architecture
model <- keras_model_sequential()
model %>% layer_dense(...)
# Compile, fit, evaluate...
# Custom prediction preprocessing
# Manual visualization
```

With easyNNR:
```r
# 😊 easyNNR approach: 1 line!
model <- easy_nn(data, target = "outcome")
```

---

## 🚀 Installation

### Step 1: Install the Package

```r
# Install devtools if you don't have it
install.packages("devtools")

# Install easyNNR from GitHub
devtools::install_github("AmeerTamoorKhan/easyNNR")
```

### Step 2: Install TensorFlow Backend

easyNNR requires TensorFlow 2.15.0 and Keras 2.15.0. Install them with one command:

```r
library(easyNNR)

# Automatic installation (recommended)
install_easyNNR_backend()

# Or specify installation method
install_easyNNR_backend(method = "conda")  # For Anaconda users
install_easyNNR_backend(method = "virtualenv")  # For virtualenv users
```

This will:
- Create a Python virtual environment
- Install TensorFlow 2.15.0
- Install Keras 2.15.0
- Configure all dependencies

### Step 3: Verify Installation

```r
library(easyNNR)

# Test with a simple example
model <- easy_nn(iris, target = "Species")

# If you see the training progress and results, you're ready! 🎉
```

### Troubleshooting Installation

**Problem: "TensorFlow backend not found"**
```r
# Solution: Reinstall backend
install_easyNNR_backend()
```

**Problem: GPU not detected**
```r
# Check CUDA availability
tensorflow::tf$config$list_physical_devices("GPU")

# easyNNR will automatically use GPU if available
```

**Problem: Reticulate configuration issues**
```r
# Check Python configuration
reticulate::py_config()

# Specify Python version (if needed)
reticulate::use_python("/path/to/python")
```

---

## ⚡ Quick Start

### Example 1: Classification (Iris Dataset)

```r
library(easyNNR)
data(iris)

# Train a neural network classifier
model <- easy_nn(
  data = iris,
  target = "Species",
  task = "classification",
  layers = c(64, 32),
  activations = "relu",
  batch_norm = TRUE,
  epochs = 50
)

# View model summary
easy_summary(model)

# Plot training history
easy_plot(model)

# Make predictions
predictions <- easy_predict(model, iris[1:10, ])
print(predictions)
```

### Example 2: Regression (mtcars Dataset)

```r
library(easyNNR)
data(mtcars)

# Train a regression model
model <- easy_nn(
  data = mtcars,
  target = "mpg",
  task = "regression",
  layers = c(128, 64),
  activations = "relu",
  epochs = 100,
  scale_data = "standard"
)

# View results
easy_summary(model)

# Plot predictions vs actual
easy_plot_regression(model)

# Predict fuel efficiency for new cars
new_cars <- mtcars[1:5, ]
mpg_predictions <- easy_predict(model, new_cars)
```

### Example 3: Real-World Wine Quality Prediction

```r
library(easyNNR)

# Load wine quality dataset
wine_red <- read.csv("winequality-red.csv", sep = ";")

# Clean dataset (optional)
wine_red <- wine_red[wine_red$quality >= 3 & wine_red$quality <= 8, ]

# Train model with advanced preprocessing
model <- easy_nn(
  data = wine_red,
  target = "quality",
  task = "regression",
  
  # Network architecture
  layers = c(128, 64),
  activations = "relu",
  dropout = 0.1,
  batch_norm = TRUE,
  
  # Training configuration
  learning_rate = 0.001,
  epochs = 60,
  batch_size = 32,
  
  # Advanced preprocessing
  preprocess = list(
    outlier_method = "winsorize",
    feature_selection = "mutual_info",
    target_transform = "log1p"
  ),
  
  scale_data = "standard",
  seed = 42
)

# Save everything
save_easyNNR_model(
  model = model,
  output_dir = "results/wine_model",
  save_plots = TRUE,
  save_logs = TRUE,
  save_best = TRUE
)

# Make predictions
sample_wines <- wine_red[1:10, ]
quality_predictions <- easy_predict(model, sample_wines)

# View results
results <- data.frame(
  Actual = sample_wines$quality,
  Predicted = round(quality_predictions, 2),
  Error = round(quality_predictions - sample_wines$quality, 2)
)
print(results)
```

---

## 📖 Complete Function Reference

### Core Training Function

#### `easy_nn()` - Train Neural Network

The main function for training neural networks.

```r
model <- easy_nn(
  data,                    # Data frame with features and target
  target,                  # Name of target column (string)
  task = NULL,             # "regression" or "classification" (auto-detected)
  exclude = NULL,          # Columns to exclude from training
  test_split = 0.2,        # Proportion for test set
  validation_split = 0.2,  # Proportion for validation set
  layers = c(128, 64),     # Hidden layer sizes
  activations = NULL,      # Activation functions
  dropout = NULL,          # Dropout rates
  batch_norm = FALSE,      # Use batch normalization
  optimizer = "adam",      # Optimizer name
  learning_rate = 0.001,   # Learning rate
  loss = NULL,             # Loss function (auto-detected)
  metrics = NULL,          # Metrics to track
  epochs = 50,             # Number of training epochs
  batch_size = 32,         # Batch size
  early_stopping = TRUE,   # Enable early stopping
  patience = 15,           # Early stopping patience
  scale_data = TRUE,       # Scaling method
  seed = 42,               # Random seed
  verbose = TRUE,          # Print progress
  preprocess = list()      # Preprocessing options (see below)
)
```

**Returns:** An `easyNNR` S3 object containing:
- `model` - Trained Keras model
- `history` - Training history (tidy format)
- `evaluation` - Test set metrics
- `predictions` - Test set predictions
- `recipe` - Preprocessing pipeline
- `parameters` - Model configuration
- `preprocessing` - Preprocessing artifacts

### Prediction Functions

#### `easy_predict()` - Standard Prediction

```r
predictions <- easy_predict(
  object,              # Trained easyNNR model
  new_data,           # Data frame with same structure as training
  type = NULL,        # "class", "prob", or "response"
  verbose = TRUE      # Print progress
)
```

**Return types:**
- **Classification**:
  - `type = "class"` - Predicted class labels (default)
  - `type = "prob"` - Probability matrix (one column per class)
- **Regression**:
  - `type = "response"` - Numeric predictions (default)

**Example:**
```r
# Get class predictions
classes <- easy_predict(model, new_data, type = "class")

# Get probabilities
probs <- easy_predict(model, new_data, type = "prob")

# Regression predictions (auto-scaled back to original)
values <- easy_predict(model, new_data)
```

#### `easy_predict_batch()` - Large Dataset Prediction

For memory-efficient prediction on large datasets:

```r
predictions <- easy_predict_batch(
  object,              # Trained easyNNR model
  new_data,           # Large data frame
  batch_size = 1000,  # Samples per batch
  type = NULL,        # Output type
  verbose = TRUE      # Show progress bar
)
```

**When to use:**
- Datasets with >100,000 rows
- Limited memory situations
- Production batch scoring

#### `easy_predict_confidence()` - Uncertainty Estimation

Get predictions with uncertainty estimates using Monte Carlo dropout:

```r
result <- easy_predict_confidence(
  object,                  # Trained easyNNR model
  new_data,               # Data to predict
  n_iterations = 100,     # Number of MC samples
  verbose = TRUE          # Print progress
)

# Returns list with:
# - predictions: Mean prediction
# - uncertainty: Standard deviation
# - iterations: All MC samples
```

**Use cases:**
- Identifying uncertain predictions
- Active learning
- Risk assessment
- Bayesian approximation

**Example:**
```r
conf <- easy_predict_confidence(model, test_data, n_iterations = 100)

# Find high-uncertainty samples
uncertain <- which(conf$uncertainty > quantile(conf$uncertainty, 0.95))

# View predictions with confidence
data.frame(
  prediction = conf$predictions,
  uncertainty = conf$uncertainty
)
```

### Visualization Functions

#### `easy_plot()` - Training History

Plot training and validation metrics over epochs:

```r
easy_plot(
  object,                      # easyNNR model
  metrics = NULL,              # Metrics to plot (NULL = all)
  smooth = FALSE,              # Add smoothed trend line
  theme = "minimal",           # "minimal", "classic", "bw", "light"
  title = NULL,                # Custom title
  subtitle = NULL              # Custom subtitle
)
```

**Example:**
```r
# Basic plot
easy_plot(model)

# Plot specific metrics with smoothing
easy_plot(model, metrics = c("loss", "accuracy"), smooth = TRUE)

# Custom theme
easy_plot(model, theme = "classic", title = "My Model Training")
```

#### `easy_plot_regression()` - Actual vs Predicted

For regression models, plot predicted vs actual values:

```r
easy_plot_regression(
  object,                  # easyNNR regression model
  title = "Regression Results"
)
```

Features:
- Scatter plot with perfect prediction line
- Color-coded residuals
- R² value displayed
- Correlation metrics

#### `easy_plot_residuals()` - Residual Analysis

Plot residuals vs fitted values:

```r
easy_plot_residuals(
  object,                  # easyNNR regression model
  title = "Residual Plot"
)
```

**Use for:**
- Detecting heteroscedasticity
- Identifying non-linearity
- Finding outliers
- Validating model assumptions

#### `easy_plot_residual_dist()` - Residual Distribution

Plot histogram of residuals:

```r
easy_plot_residual_dist(
  object,                  # easyNNR regression model
  bins = 30,              # Number of histogram bins
  title = "Residual Distribution"
)
```

**Checks for:**
- Normality of residuals
- Systematic bias
- Outliers

#### `easy_plot_confusion()` - Confusion Matrix

For classification models:

```r
easy_plot_confusion(
  object,                  # easyNNR classification model
  normalize = TRUE,        # Show percentages or counts
  title = "Confusion Matrix"
)
```

**Example:**
```r
# Show percentages (default)
easy_plot_confusion(model)

# Show raw counts
easy_plot_confusion(model, normalize = FALSE)
```

#### `easy_plot_class_distribution()` - Class Balance

Visualize class distribution in training and predictions:

```r
easy_plot_class_distribution(
  object,                  # easyNNR classification model
  title = "Class Distribution"
)
```

**Use for:**
- Detecting class imbalance
- Comparing training vs prediction distribution
- Identifying bias

#### `easy_plot_importance()` - Feature Importance

Plot feature importance scores:

```r
easy_plot_importance(
  object,                  # easyNNR model
  n_features = 20,         # Number of top features
  method = "permutation",  # Importance method
  title = "Feature Importance"
)
```

**Methods:**
- `"permutation"` - Permutation importance (default)
- `"gradient"` - Gradient-based importance
- `"weight"` - Weight magnitude importance

### Model Summary and Information

#### `easy_summary()` - Detailed Model Report

```r
easy_summary(object)  # easyNNR model
```

**Displays:**
- Task type and dataset size
- Network architecture (layer by layer)
- Training configuration
- Preprocessing steps applied
- Evaluation metrics
- Feature selection results
- Training time and epochs

#### `print()` - Quick Model Overview

```r
print(object)  # easyNNR model
# or just
model
```

**Shows:**
- Model type and architecture
- Dataset split sizes
- Key performance metrics
- Training status

#### `summary()` - S3 Method

```r
summary(object)  # Same as easy_summary()
```

### Model Persistence

#### `save_easyNNR_model()` - Save Complete Model

Save model with all components for later use:

```r
save_easyNNR_model(
  model,                   # easyNNR model object
  output_dir,             # Directory to save to
  save_plots = TRUE,      # Save visualization plots
  save_logs = TRUE,       # Save training summary
  save_best = TRUE        # Save best model snapshot
)
```

**Creates directory structure:**
```
output_dir/
├── models/
│   ├── model_weights.h5          # Neural network weights
│   ├── model_architecture.rds    # Model configuration
│   ├── preprocessing.rds          # Preprocessing pipeline
│   └── best_model_0.87_20251127/ # Best model snapshot (optional)
├── plots/
│   ├── training_history.png
│   ├── actual_vs_predicted.png   # Regression only
│   ├── confusion_matrix.png      # Classification only
│   ├── residuals.png
│   └── residual_distribution.png
├── logs/
│   └── summary_report.txt        # Complete training log
├── metrics_20251127_143022.csv
├── training_history_20251127_143022.csv
└── test_predictions_20251127_143022.csv
```

**Example:**
```r
# Save everything
save_easyNNR_model(
  model = model,
  output_dir = "results/my_model",
  save_plots = TRUE,
  save_logs = TRUE,
  save_best = TRUE
)

# Minimal save (just model)
save_easyNNR_model(
  model = model,
  output_dir = "results/my_model",
  save_plots = FALSE,
  save_logs = FALSE,
  save_best = FALSE
)
```

#### `load_easyNNR_model()` - Load Saved Model

Load a previously saved model:

```r
model <- load_easyNNR_model(
  path  # Path to model_architecture.rds file
)
```

**Example:**
```r
# Load model
loaded_model <- load_easyNNR_model(
  "results/my_model/models/model_architecture.rds"
)

# Model is fully functional
predictions <- easy_predict(loaded_model, new_data)
easy_plot(loaded_model)
```

**What gets loaded:**
- Complete neural network with weights
- All preprocessing transformations
- Feature selection configuration
- Target transformations (regression)
- Class mappings (classification)

**Cross-session compatibility:** Models saved in one R session can be loaded in another, even after restarting R.

### Logging System

#### `start_easyNNR_log()` - Begin Logging

Capture all console output to file:

```r
start_easyNNR_log(
  log_dir  # Directory to save log files
)
```

#### `end_easyNNR_log()` - Stop Logging

Stop logging and close file:

```r
end_easyNNR_log()
```

**Example workflow:**
```r
# Start logging
start_easyNNR_log("logs/")

# All output is now captured
model <- easy_nn(iris, target = "Species")
easy_summary(model)
predictions <- easy_predict(model, iris[1:10, ])

# Stop logging
end_easyNNR_log()

# Creates: logs/easyNNR_log_20251127_143022.txt
```

**Log file includes:**
- Timestamp
- Complete training progress
- All warnings and messages
- Model summaries
- Prediction results

### Utility Functions

#### `easy_compare()` - Compare Multiple Models

Compare performance of different models:

```r
easy_compare(
  model1, model2, model3,        # easyNNR models
  metrics = c("accuracy", "f1")  # Metrics to compare
)
```

#### `easy_dashboard()` - Interactive Dashboard

Launch interactive Shiny dashboard:

```r
easy_dashboard(model)
```

**Features:**
- Real-time predictions
- Interactive plots
- Feature importance explorer
- Model diagnostics

---

## 🎛️ Training Parameters Guide

### Data Input Parameters

#### `data` (required)
**Type:** `data.frame` or `tibble`  
**Description:** Your dataset containing features and target variable

**Requirements:**
- Must have at least 2 columns
- Cannot be empty
- Can contain numeric, character, or factor columns
- Missing values allowed (will be imputed)

**Example:**
```r
# From built-in dataset
data(iris)
model <- easy_nn(data = iris, target = "Species")

# From CSV file
df <- read.csv("mydata.csv")
model <- easy_nn(data = df, target = "outcome")

# From tibble
library(dplyr)
df <- tibble(x1 = rnorm(100), x2 = rnorm(100), y = rbinom(100, 1, 0.5))
model <- easy_nn(data = df, target = "y")
```

#### `target` (required)
**Type:** `character` (column name)  
**Description:** Name of the column to predict

**Example:**
```r
model <- easy_nn(data = iris, target = "Species")
model <- easy_nn(data = mtcars, target = "mpg")
```

#### `task`
**Type:** `character` or `NULL`  
**Options:** `"regression"`, `"classification"`, `NULL` (auto-detect)  
**Default:** `NULL`

**Auto-detection rules:**
- Factor or character target → classification
- Numeric with >10 unique values → regression
- Integer with ≤20 unique values → classification

**Example:**
```r
# Auto-detect (recommended)
model <- easy_nn(data = iris, target = "Species")

# Explicit specification
model <- easy_nn(data = iris, target = "Species", task = "classification")
model <- easy_nn(data = mtcars, target = "mpg", task = "regression")
```

#### `exclude`
**Type:** `character` vector or `NULL`  
**Default:** `NULL`  
**Description:** Column names to exclude from training (e.g., IDs, timestamps)

**Example:**
```r
model <- easy_nn(
  data = customer_data,
  target = "churned",
  exclude = c("customer_id", "signup_date", "last_login")
)
```

**Use cases:**
- Unique identifiers
- Date columns (unless using date feature extraction)
- Text columns (requires separate preprocessing)
- Columns with data leakage

---

### Data Splitting Parameters

#### `test_split`
**Type:** `numeric`  
**Default:** `0.2` (20%)  
**Range:** `0.0` to `1.0`  
**Description:** Proportion of data reserved for final testing

**Example:**
```r
# Use 30% for testing
model <- easy_nn(data = df, target = "y", test_split = 0.3)

# Use 10% for testing (more training data)
model <- easy_nn(data = df, target = "y", test_split = 0.1)

# No test set (not recommended)
model <- easy_nn(data = df, target = "y", test_split = 0.0)
```

**Guidelines:**
- Small datasets (<1000 rows): 0.2-0.3
- Medium datasets (1000-10000): 0.2
- Large datasets (>10000): 0.1-0.15

#### `validation_split`
**Type:** `numeric`  
**Default:** `0.2` (20% of training data)  
**Range:** `0.0` to `1.0`  
**Description:** Proportion of training data for validation (early stopping)

**Example:**
```r
# Use 25% of training data for validation
model <- easy_nn(
  data = df,
  target = "y",
  validation_split = 0.25
)

# No validation (disables early stopping monitoring)
model <- easy_nn(
  data = df,
  target = "y",
  validation_split = 0.0,
  early_stopping = FALSE
)
```

**Note:** Validation data is taken from training data AFTER test split.

**Effective splits:**
```r
test_split = 0.2, validation_split = 0.2
# Result:
# - Training: 64% (0.8 * 0.8)
# - Validation: 16% (0.8 * 0.2)
# - Test: 20%
```

---

### Network Architecture Parameters

#### `layers`
**Type:** `integer` vector  
**Default:** `c(128, 64)`  
**Description:** Number of neurons in each hidden layer

**Examples:**
```r
# Shallow network (faster, less capacity)
layers = c(64)

# Default (balanced)
layers = c(128, 64)

# Deep network (more capacity, slower)
layers = c(256, 128, 64, 32)

# Very deep
layers = c(512, 256, 128, 64, 32, 16)

# Wide and shallow
layers = c(512, 256)
```

**Guidelines:**
- **Simple problems:** 1-2 layers, 32-128 neurons
- **Medium complexity:** 2-3 layers, 64-256 neurons
- **Complex problems:** 3-5 layers, 128-512 neurons
- **Start small** and increase if underfitting

#### `activations`
**Type:** `character` or `character` vector  
**Default:** `NULL` (uses "relu" for all layers)  
**Options:** `"relu"`, `"leaky_relu"`, `"elu"`, `"selu"`, `"sigmoid"`, `"tanh"`, `"softmax"`

**Single activation (all layers):**
```r
activations = "relu"  # All layers use ReLU
activations = "elu"   # All layers use ELU
```

**Per-layer activations:**
```r
# Different activation per layer
layers = c(128, 64, 32)
activations = c("relu", "relu", "sigmoid")
```

**Activation guide:**
- **ReLU** (`"relu"`): Default choice, fast, works well
- **Leaky ReLU** (`"leaky_relu"`): Prevents dying neurons
- **ELU** (`"elu"`): Smooth, can be faster to converge
- **SELU** (`"selu"`): Self-normalizing (use with specific initialization)
- **Sigmoid** (`"sigmoid"`): Output layer for binary classification
- **Tanh** (`"tanh"`): Alternative to sigmoid, zero-centered
- **Softmax**: Automatically added for multiclass classification

**Example:**
```r
# Deep network with varied activations
model <- easy_nn(
  data = df,
  target = "y",
  layers = c(256, 128, 64, 32),
  activations = c("relu", "relu", "elu", "elu")
)
```

#### `dropout`
**Type:** `numeric` or `numeric` vector or `NULL`  
**Default:** `NULL` (no dropout)  
**Range:** `0.0` to `1.0` (typically 0.1-0.5)  
**Description:** Dropout rate for regularization

**Single dropout (all layers):**
```r
dropout = 0.2  # 20% dropout on all layers
dropout = 0.5  # 50% dropout (aggressive regularization)
```

**Per-layer dropout:**
```r
layers = c(256, 128, 64)
dropout = c(0.4, 0.3, 0.2)  # Decreasing dropout schedule
```

**Guidelines:**
- **No overfitting:** `dropout = NULL` or `dropout = 0`
- **Mild overfitting:** `dropout = 0.1-0.2`
- **Strong overfitting:** `dropout = 0.3-0.5`
- **Very strong overfitting:** `dropout = 0.5-0.7`

**Example:**
```r
# Regularization schedule: more dropout in early layers
model <- easy_nn(
  data = df,
  target = "y",
  layers = c(512, 256, 128, 64),
  dropout = c(0.5, 0.4, 0.3, 0.2)
)
```

#### `batch_norm`
**Type:** `logical`  
**Default:** `FALSE`  
**Description:** Whether to use batch normalization between layers

**Example:**
```r
# Enable batch normalization
model <- easy_nn(
  data = df,
  target = "y",
  batch_norm = TRUE
)
```

**Benefits:**
- Faster training
- More stable training
- Can use higher learning rates
- Acts as regularization

**When to use:**
- Deep networks (>3 layers)
- Training instability
- Slow convergence

**When not to use:**
- Very small batches (<16)
- Simple shallow networks
- When overfitting is already strong

---

### Optimization Parameters

#### `optimizer`
**Type:** `character` or keras optimizer object  
**Default:** `"adam"`  
**Options:** `"adam"`, `"adamw"`, `"rmsprop"`, `"sgd"`, `"adagrad"`, `"adadelta"`, `"nadam"`

**String options:**
```r
optimizer = "adam"      # Adaptive Moment Estimation (default, best general choice)
optimizer = "adamw"     # Adam with Weight Decay (better generalization)
optimizer = "rmsprop"   # RMSProp (good for RNNs)
optimizer = "sgd"       # Stochastic Gradient Descent (classic)
optimizer = "adagrad"   # Adaptive Gradient (good for sparse data)
optimizer = "adadelta"  # Adaptive Delta
optimizer = "nadam"     # Nesterov Adam
```

**Custom optimizer:**
```r
custom_opt <- keras::optimizer_adam(
  learning_rate = 0.001,
  beta_1 = 0.95,
  beta_2 = 0.999,
  epsilon = 1e-07
)

model <- easy_nn(
  data = df,
  target = "y",
  optimizer = custom_opt
)
```

**Optimizer guide:**
- **adam**: Default choice, adaptive learning rate, works well for most problems
- **adamw**: Similar to adam but better generalization (recommended for complex models)
- **rmsprop**: Good for recurrent networks and non-stationary objectives
- **sgd**: Classic, may need learning rate scheduling, good for finding sharp minima
- **nadam**: Nesterov momentum + Adam, can converge faster

#### `learning_rate`
**Type:** `numeric`  
**Default:** `0.001`  
**Typical range:** `0.0001` to `0.01`

**Examples:**
```r
learning_rate = 0.001   # Default
learning_rate = 0.0001  # Conservative (very stable)
learning_rate = 0.01    # Aggressive (faster but risky)
learning_rate = 0.0005  # Fine-tuning
```

**Guidelines:**
- **Too high:** Training unstable, loss oscillates or increases
- **Too low:** Training very slow, may not converge
- **Start at 0.001** and adjust based on training curves
- **Use lower rates (0.0001-0.0005)** for fine-tuning
- **Use higher rates (0.005-0.01)** for initial exploration

**Adaptive learning:**
```r
# Start high, then decay manually
model1 <- easy_nn(data = df, target = "y", learning_rate = 0.01, epochs = 20)
model2 <- easy_nn(data = df, target = "y", learning_rate = 0.001, epochs = 30)
```

#### `loss`
**Type:** `character` or `NULL`  
**Default:** `NULL` (auto-detected)

**Auto-detection:**
- **Regression:** `"mse"` (mean squared error)
- **Binary classification:** `"binary_crossentropy"`
- **Multiclass classification:** `"categorical_crossentropy"` or `"sparse_categorical_crossentropy"`

**Manual specification:**
```r
# Regression losses
loss = "mse"                    # Mean Squared Error
loss = "mae"                    # Mean Absolute Error (robust to outliers)
loss = "huber"                  # Huber loss (robust)
loss = "mean_squared_logarithmic_error"  # MSLE (for large value ranges)

# Classification losses
loss = "binary_crossentropy"    # Binary classification
loss = "categorical_crossentropy"  # Multiclass (one-hot encoded)
loss = "sparse_categorical_crossentropy"  # Multiclass (integer labels)
```

**Example:**
```r
# Use MAE for regression (robust to outliers)
model <- easy_nn(
  data = df,
  target = "price",
  task = "regression",
  loss = "mae"
)
```

#### `metrics`
**Type:** `character` vector or `NULL`  
**Default:** `NULL` (auto-selected based on task)

**Auto-selected metrics:**
- **Regression:** `c("mae", "mse")`
- **Classification:** `c("accuracy")`

**Available metrics:**
```r
# Regression
metrics = c("mae", "mse", "rmse", "mape")

# Binary classification
metrics = c("accuracy", "binary_accuracy", "precision", "recall")

# Multiclass classification
metrics = c("accuracy", "categorical_accuracy", "top_k_categorical_accuracy")
```

**Example:**
```r
# Track multiple metrics
model <- easy_nn(
  data = df,
  target = "y",
  task = "regression",
  metrics = c("mae", "mse", "mape")
)

# Classification with precision and recall
model <- easy_nn(
  data = df,
  target = "class",
  task = "classification",
  metrics = c("accuracy", "precision", "recall")
)
```

---

### Training Control Parameters

#### `epochs`
**Type:** `integer`  
**Default:** `50`  
**Description:** Maximum number of training iterations through the dataset

**Guidelines:**
- **Simple problems:** 20-50 epochs
- **Medium complexity:** 50-100 epochs
- **Complex problems:** 100-500 epochs
- **With early stopping:** Set high (100-200), let early stopping decide

**Example:**
```r
# Short training
model <- easy_nn(data = df, target = "y", epochs = 30)

# Long training with early stopping
model <- easy_nn(
  data = df,
  target = "y",
  epochs = 200,
  early_stopping = TRUE,
  patience = 20
)
```

#### `batch_size`
**Type:** `integer`  
**Default:** `32`  
**Typical range:** `16` to `256`

**Guidelines:**
- **Small datasets (<1000):** 16-32
- **Medium datasets (1000-10000):** 32-64
- **Large datasets (>10000):** 64-256
- **GPU available:** Larger batches (128-256)
- **Limited memory:** Smaller batches (16-32)

**Trade-offs:**
- **Larger batches:** Faster training, more stable gradients, more memory
- **Smaller batches:** Better generalization, less memory, more noise in gradients

**Example:**
```r
# Small batch (better generalization)
model <- easy_nn(data = df, target = "y", batch_size = 16)

# Large batch (faster training)
model <- easy_nn(data = df, target = "y", batch_size = 128)
```

#### `early_stopping`
**Type:** `logical`  
**Default:** `TRUE`  
**Description:** Stop training when validation loss stops improving

**Example:**
```r
# With early stopping (recommended)
model <- easy_nn(
  data = df,
  target = "y",
  early_stopping = TRUE,
  patience = 15
)

# Without early stopping
model <- easy_nn(
  data = df,
  target = "y",
  early_stopping = FALSE,
  epochs = 50
)
```

#### `patience`
**Type:** `integer`  
**Default:** `15`  
**Description:** Number of epochs to wait for improvement before stopping

**Guidelines:**
- **Fast convergence:** `patience = 10-15`
- **Slow convergence:** `patience = 20-30`
- **Very noisy validation:** `patience = 30-50`

**Example:**
```r
# Patient training
model <- easy_nn(
  data = df,
  target = "y",
  early_stopping = TRUE,
  patience = 25  # Wait 25 epochs without improvement
)
```

---

### Data Preprocessing Parameters

#### `scale_data`
**Type:** `logical` or `character`  
**Default:** `TRUE` (standard scaling)  
**Options:** `TRUE`, `FALSE`, `"standard"`, `"minmax"`, `"robust"`, `"maxabs"`, `"quantile"`

**Scaling methods:**

**Standard (Z-score normalization):**
```r
scale_data = TRUE
# or
scale_data = "standard"
# Formula: (x - mean) / sd
# Result: Mean = 0, SD = 1
```

**Min-Max normalization:**
```r
scale_data = "minmax"
# Formula: (x - min) / (max - min)
# Result: Range [0, 1]
```

**Robust scaling:**
```r
scale_data = "robust"
# Formula: (x - median) / IQR
# Result: Robust to outliers
```

**Max-Abs scaling:**
```r
scale_data = "maxabs"
# Formula: x / max(abs(x))
# Result: Range [-1, 1], preserves sparsity
```

**Quantile transformation:**
```r
scale_data = "quantile"
# Maps to uniform or normal distribution
# Result: Non-linear transformation
```

**No scaling:**
```r
scale_data = FALSE
# Use when:
# - Features already scaled
# - Tree-based ensembles
# - Custom preprocessing
```

**Guidelines:**
- **Most cases:** Use `"standard"` (default)
- **Bounded outputs needed:** Use `"minmax"`
- **Heavy outliers:** Use `"robust"`
- **Sparse data:** Use `"maxabs"`

#### `preprocess`
**Type:** `list`  
**Default:** `list()` (no advanced preprocessing)  
**Description:** Advanced preprocessing options (see next section)

---

## 🔧 Preprocessing Options

The `preprocess` parameter accepts a list of advanced preprocessing options. All options are optional with sensible defaults.

### Outlier Handling

Control how outliers are detected and handled.

```r
preprocess = list(
  outlier_method = "winsorize",     # Detection/handling method
  outlier_threshold = 0.05          # Threshold for detection
)
```

**Available methods:**

**`outlier_method`:**
- `"none"` - No outlier handling (default)
- `"iqr"` - Interquartile range method (threshold: 1.5)
- `"zscore"` - Z-score method (threshold: 3 standard deviations)
- `"isolation_forest"` - Percentile-based approximation (threshold: 0.1)
- `"winsorize"` - Cap at percentiles (threshold: 0.05 = 5th/95th percentile)
- `"cap"` - Hard cap at IQR boundaries (threshold: 1.5)
- `"remove"` - Remove outlier rows completely

**Examples:**

```r
# IQR method (1.5 * IQR outside Q1/Q3)
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    outlier_method = "iqr",
    outlier_threshold = 1.5
  )
)

# Z-score method (3 standard deviations)
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    outlier_method = "zscore",
    outlier_threshold = 3
  )
)

# Winsorize at 5th and 95th percentiles
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    outlier_method = "winsorize",
    outlier_threshold = 0.05
  )
)

# Remove extreme outliers
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    outlier_method = "remove",
    outlier_threshold = 3  # 3 SD from mean
  )
)
```

**When to use each:**
- **IQR:** General purpose, robust
- **Z-score:** When outliers are rare and extreme
- **Winsorize:** Preserve sample size, reduce extreme values
- **Cap:** Similar to winsorize with IQR
- **Remove:** When outliers are truly erroneous data

### Target Transformation (Regression Only)

Transform the target variable to improve model performance.

```r
preprocess = list(
  target_transform = "log1p"  # Transformation method
)
```

**Available transformations:**
- `"none"` - No transformation (default)
- `"log"` - Natural logarithm (requires positive values)
- `"log1p"` - log(1 + x) (handles zeros)
- `"sqrt"` - Square root (requires non-negative values)
- `"boxcox"` - Box-Cox power transformation (estimates optimal lambda)
- `"yeojohnson"` - Yeo-Johnson transformation (handles negative values)
- `"quantile"` - Quantile normalization

**Examples:**

```r
# Log transform for right-skewed targets
model <- easy_nn(
  data = housing_data,
  target = "price",
  preprocess = list(
    target_transform = "log"
  )
)

# Log1p for targets with zeros
model <- easy_nn(
  data = count_data,
  target = "count",
  preprocess = list(
    target_transform = "log1p"
  )
)

# Box-Cox (auto-finds best transformation)
model <- easy_nn(
  data = df,
  target = "sales",
  preprocess = list(
    target_transform = "boxcox"
  )
)

# Yeo-Johnson (handles negative values)
model <- easy_nn(
  data = df,
  target = "profit",  # Can be negative
  preprocess = list(
    target_transform = "yeojohnson"
  )
)
```

**Automatic inverse transformation:**
All predictions are automatically transformed back to the original scale:
```r
# Model trained with log1p transform
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(target_transform = "log1p")
)

# Predictions automatically in original scale
predictions <- easy_predict(model, new_data)
# No need to manually apply exp(pred) - 1
```

**When to use:**
- **Heavy right skew:** `"log"` or `"log1p"`
- **Non-constant variance:** `"boxcox"` or `"yeojohnson"`
- **Negative values:** `"yeojohnson"` or `"quantile"`
- **Need normality:** `"boxcox"`, `"yeojohnson"`, or `"quantile"`

### Class Imbalance Handling (Classification Only)

Handle imbalanced class distributions.

```r
preprocess = list(
  imbalance_method = "smote",  # Resampling method
  imbalance_ratio = 1.0        # Target minority/majority ratio
)
```

**Available methods:**
- `"none"` - No resampling (default)
- `"oversample"` - Random oversampling of minority classes
- `"undersample"` - Random undersampling of majority class
- `"smote"` - Synthetic Minority Over-sampling Technique
- `"adasyn"` - Adaptive Synthetic Sampling
- `"class_weights"` - Balanced loss weighting (no resampling)

**Parameters:**
- `imbalance_ratio`: Target ratio of minority to majority class
  - `1.0` = Perfect balance (default)
  - `0.5` = Half as many minority as majority
  - `0.8` = 80% balance

**Examples:**

```r
# SMOTE (synthetic oversampling)
model <- easy_nn(
  data = imbalanced_data,
  target = "fraud",  # 1% fraud, 99% normal
  preprocess = list(
    imbalance_method = "smote",
    imbalance_ratio = 1.0  # Fully balance classes
  )
)

# ADASYN (adaptive synthetic sampling)
model <- easy_nn(
  data = imbalanced_data,
  target = "disease",
  preprocess = list(
    imbalance_method = "adasyn",
    imbalance_ratio = 0.8  # 80% balance
  )
)

# Class weights (no resampling, just loss weighting)
model <- easy_nn(
  data = imbalanced_data,
  target = "churn",
  preprocess = list(
    imbalance_method = "class_weights"
  )
)

# Undersample majority class
model <- easy_nn(
  data = imbalanced_data,
  target = "rare_event",
  preprocess = list(
    imbalance_method = "undersample",
    imbalance_ratio = 1.0
  )
)
```

**Method comparison:**
- **class_weights:** Fastest, no data change, good first choice
- **smote:** Generates synthetic samples, improves decision boundary
- **adasyn:** Like SMOTE but focuses on harder-to-learn regions
- **oversample:** Simple, may overfit on duplicates
- **undersample:** Loses majority class information

**When to use:**
- **Mild imbalance (70:30):** Try `class_weights` first
- **Moderate imbalance (90:10):** `smote` or `adasyn`
- **Severe imbalance (>95:5):** Combination of `smote` + `class_weights`
- **Large dataset:** `undersample` to reduce size

### Feature Selection

Reduce dimensionality by selecting most informative features.

```r
preprocess = list(
  feature_selection = "mutual_info",  # Selection method
  n_features = 20,                   # Number to keep
  correlation_threshold = 0.9        # For correlation method
)
```

**Available methods:**
- `"none"` - No feature selection (default)
- `"variance"` - Low variance filter
- `"correlation"` - Remove highly correlated features
- `"mutual_info"` - Mutual information scoring
- `"rfe"` - Recursive Feature Elimination
- `"lasso"` - L1 regularization-based selection

**Parameters:**
- `n_features`: Number of features to keep (NULL = auto-determine)
- `correlation_threshold`: Threshold for correlation method (default: 0.9)

**Examples:**

```r
# Mutual information (recommended)
model <- easy_nn(
  data = high_dim_data,  # 100 features
  target = "y",
  preprocess = list(
    feature_selection = "mutual_info",
    n_features = 30  # Keep top 30
  )
)

# Remove highly correlated features
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    feature_selection = "correlation",
    correlation_threshold = 0.85  # Remove if r > 0.85
  )
)

# LASSO-based selection
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    feature_selection = "lasso",
    n_features = 25
  )
)

# RFE (slower but thorough)
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    feature_selection = "rfe",
    n_features = 15
  )
)

# Low variance filter
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    feature_selection = "variance"
  )
)
```

**When to use each:**
- **mutual_info:** Best general choice, works for any task
- **correlation:** Fast, removes redundancy
- **lasso:** Good for linear relationships
- **rfe:** Most thorough but slowest
- **variance:** Quick initial filter

**Benefits:**
- Faster training
- Reduced overfitting
- Improved interpretability
- Better generalization

### Categorical Encoding

Choose how to encode categorical variables.

```r
preprocess = list(
  encoding = "onehot",        # Encoding method
  max_categories = 50         # Max categories for one-hot
)
```

**Available methods:**
- `"onehot"` - One-hot encoding (default)
- `"target"` - Target/mean encoding
- `"frequency"` - Frequency-based encoding
- `"binary"` - Binary encoding
- `"hash"` - Feature hashing

**Examples:**

```r
# One-hot encoding (default)
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    encoding = "onehot",
    max_categories = 50  # Error if >50 categories
  )
)

# Target encoding (for high cardinality)
model <- easy_nn(
  data = customer_data,
  target = "purchase",
  preprocess = list(
    encoding = "target"  # Encode by mean of target
  )
)

# Frequency encoding
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    encoding = "frequency"  # Encode by frequency
  )
)

# Hash encoding (for very high cardinality)
model <- easy_nn(
  data = text_data,
  target = "category",
  preprocess = list(
    encoding = "hash"
  )
)
```

**When to use each:**
- **onehot:** Low cardinality (<50 categories), standard choice
- **target:** High cardinality, captures relationship with target
- **frequency:** Ordinal relationship with frequency
- **binary:** Moderate cardinality (10-100 categories)
- **hash:** Very high cardinality (>100 categories)

**Warning:** Target encoding can cause overfitting if not careful. Best used with cross-validation.

### Imputation

Handle missing values in features.

```r
preprocess = list(
  impute_numeric = "median",       # Numeric imputation
  impute_categorical = "mode",     # Categorical imputation
  add_indicators = FALSE           # Add missingness flags
)
```

**Numeric imputation methods:**
- `"median"` - Median imputation (default, robust)
- `"mean"` - Mean imputation
- `"knn"` - K-nearest neighbors imputation
- `"iterative"` - MICE-style iterative imputation
- `"constant"` - Fill with constant value

**Categorical imputation methods:**
- `"mode"` - Most frequent value (default)
- `"constant"` - Fill with constant
- `"missing_category"` - Create "missing" category

**Examples:**

```r
# Robust imputation
model <- easy_nn(
  data = messy_data,
  target = "y",
  preprocess = list(
    impute_numeric = "median",
    impute_categorical = "mode"
  )
)

# Advanced imputation with indicators
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    impute_numeric = "knn",
    impute_categorical = "missing_category",
    add_indicators = TRUE  # Add is_missing_X columns
  )
)

# Iterative imputation (slow but accurate)
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    impute_numeric = "iterative"
  )
)

# Simple constant imputation
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    impute_numeric = "constant",  # Fill with 0
    impute_categorical = "constant"  # Fill with "unknown"
  )
)
```

**Missingness indicators:**
```r
add_indicators = TRUE
# Creates binary columns: is_missing_feature1, is_missing_feature2, ...
# Useful when missingness is informative
```

**When to use:**
- **Few missing (<5%):** `median`/`mode` is fine
- **Many missing (5-20%):** Consider `knn` or `iterative`
- **Missing informative:** Use `add_indicators = TRUE`
- **Quick and dirty:** `mean`/`constant`

### Feature Engineering

Create new features from existing ones.

```r
preprocess = list(
  interactions = TRUE,         # Create interaction terms
  polynomial_degree = 2,       # Polynomial features
  pca_components = 20         # PCA dimensionality reduction
)
```

**Polynomial features:**
```r
# Create polynomial and interaction terms
polynomial_degree = 1  # No polynomial (default)
polynomial_degree = 2  # Add x^2 and x1*x2 terms
polynomial_degree = 3  # Add x^3, x^2, x1*x2*x3 terms
```

**Example:**
```r
# Add polynomial features
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    polynomial_degree = 2  # x, x^2, x1*x2
  )
)
```

**PCA (Principal Component Analysis):**
```r
# Reduce to top N principal components
pca_components = NULL  # No PCA (default)
pca_components = 20    # Keep top 20 components
pca_components = 0.95  # Keep components explaining 95% variance
```

**Example:**
```r
# Dimensionality reduction
model <- easy_nn(
  data = high_dim_data,
  target = "y",
  preprocess = list(
    pca_components = 50  # Reduce from 200 to 50 features
  )
)
```

**Interactions:**
```r
interactions = FALSE         # No interactions (default)
interactions = TRUE          # Auto-generate interactions
interactions = ~ x1:x2 + x3:x4  # Specific interactions
```

**Example:**
```r
# Interaction terms
model <- easy_nn(
  data = df,
  target = "y",
  preprocess = list(
    interactions = TRUE  # Create all pairwise interactions
  )
)
```

### Time Series Features

Extract features from date columns and create lag/rolling features.

```r
preprocess = list(
  date_features = TRUE,                    # Extract from dates
  lag_features = c(1, 7, 30),             # Lag periods
  rolling_features = list(                 # Rolling statistics
    mean_7 = list(window = 7, fun = mean),
    std_30 = list(window = 30, fun = sd)
  )
)
```

**Date feature extraction:**
```r
date_features = TRUE
# Automatically extracts:
# - year, month, day
# - day_of_week, day_of_year
# - quarter
# - is_weekend, is_month_end
# - Cyclical: sin/cos of month, day
```

**Example:**
```r
# Date features
model <- easy_nn(
  data = sales_data,  # Has 'date' column
  target = "revenue",
  preprocess = list(
    date_features = TRUE
  )
)
```

**Lag features:**
```r
lag_features = c(1, 7, 30)
# Creates: feature_lag1, feature_lag7, feature_lag30
```

**Example:**
```r
# Time series with lags
model <- easy_nn(
  data = stock_data,
  target = "price",
  preprocess = list(
    lag_features = c(1, 2, 3, 5, 10)  # Multiple lags
  )
)
```

**Rolling features:**
```r
rolling_features = list(
  mean_7 = list(window = 7, fun = mean),
  std_7 = list(window = 7, fun = sd),
  max_30 = list(window = 30, fun = max),
  min_30 = list(window = 30, fun = min)
)
```

**Example:**
```r
# Complete time series preprocessing
model <- easy_nn(
  data = timeseries_data,
  target = "demand",
  preprocess = list(
    date_features = TRUE,
    lag_features = c(1, 7, 14, 30),
    rolling_features = list(
      mean_7 = list(window = 7, fun = mean),
      mean_30 = list(window = 30, fun = mean),
      std_7 = list(window = 7, fun = sd),
      max_7 = list(window = 7, fun = max),
      min_7 = list(window = 7, fun = min)
    )
  )
)
```

### Complete Preprocessing Example

Combining multiple preprocessing options:

```r
model <- easy_nn(
  data = complex_dataset,
  target = "outcome",
  task = "classification",
  
  # Network architecture
  layers = c(256, 128, 64),
  activations = "relu",
  dropout = c(0.3, 0.2, 0.1),
  batch_norm = TRUE,
  
  # Training
  learning_rate = 0.0005,
  epochs = 100,
  batch_size = 64,
  
  # Comprehensive preprocessing
  preprocess = list(
    # Data quality
    outlier_method = "winsorize",
    outlier_threshold = 0.05,
    
    # Class imbalance
    imbalance_method = "adasyn",
    imbalance_ratio = 0.8,
    
    # Feature selection
    feature_selection = "mutual_info",
    n_features = 50,
    
    # Encoding
    encoding = "target",
    max_categories = 100,
    
    # Imputation
    impute_numeric = "iterative",
    impute_categorical = "mode",
    add_indicators = TRUE,
    
    # Feature engineering
    polynomial_degree = 2,
    pca_components = 30,
    
    # Time series (if applicable)
    date_features = TRUE,
    lag_features = c(1, 7, 30),
    rolling_features = list(
      mean_7 = list(window = 7, fun = mean),
      std_30 = list(window = 30, fun = sd)
    )
  ),
  
  scale_data = "standard",
  seed = 42
)
```

---

## 🔬 Advanced Examples

### Example 1: Customer Churn Prediction

```r
library(easyNNR)

# Load customer data
customers <- read.csv("customer_data.csv")

# Full pipeline with imbalance handling
churn_model <- easy_nn(
  data = customers,
  target = "churned",
  task = "classification",
  
  # Exclude ID columns
  exclude = c("customer_id", "signup_date"),
  
  # Deep architecture
  layers = c(256, 128, 64, 32),
  activations = "relu",
  dropout = c(0.4, 0.3, 0.2, 0.1),
  batch_norm = TRUE,
  
  # Optimization
  optimizer = "adamw",
  learning_rate = 0.0005,
  epochs = 150,
  early_stopping = TRUE,
  patience = 25,
  
  # Handle imbalance (churn is typically <10%)
  preprocess = list(
    outlier_method = "winsorize",
    imbalance_method = "adasyn",
    imbalance_ratio = 0.7,  # Don't fully balance
    feature_selection = "mutual_info",
    n_features = 30,
    encoding = "target",
    date_features = TRUE
  ),
  
  scale_data = "standard",
  seed = 42
)

# Save complete model
save_easyNNR_model(
  churn_model,
  output_dir = "models/churn_model",
  save_plots = TRUE,
  save_logs = TRUE,
  save_best = TRUE
)

# Analyze results
easy_summary(churn_model)
easy_plot(churn_model)
easy_plot_confusion(churn_model)
easy_plot_importance(churn_model, n_features = 15)

# Score new customers
new_customers <- read.csv("new_customers.csv")
churn_prob <- easy_predict(churn_model, new_customers, type = "prob")

# Get confidence scores
conf_results <- easy_predict_confidence(
  churn_model, 
  new_customers, 
  n_iterations = 100
)

# Flag high-risk customers
high_risk <- new_customers[churn_prob[, "Yes"] > 0.7, ]
```

### Example 2: House Price Prediction

```r
library(easyNNR)

# Load housing data
houses <- read.csv("housing.csv")

# Regression with target transformation
price_model <- easy_nn(
  data = houses,
  target = "sale_price",
  task = "regression",
  
  # Architecture
  layers = c(512, 256, 128, 64),
  activations = c("relu", "relu", "elu", "elu"),
  dropout = 0.3,
  batch_norm = TRUE,
  
  # Training
  learning_rate = 0.0001,
  epochs = 200,
  batch_size = 64,
  
  # Advanced preprocessing
  preprocess = list(
    outlier_method = "winsorize",
    outlier_threshold = 0.02,  # Remove extreme 2%
    target_transform = "log",  # Log of price
    feature_selection = "lasso",
    n_features = 40,
    encoding = "target",
    polynomial_degree = 2,
    date_features = TRUE
  ),
  
  scale_data = "robust",  # Robust to outliers
  seed = 42
)

# Evaluate
easy_summary(price_model)
easy_plot(price_model)
easy_plot_regression(price_model)
easy_plot_residuals(price_model)
easy_plot_residual_dist(price_model)

# Predict new houses
new_houses <- read.csv("new_listings.csv")
predicted_prices <- easy_predict(price_model, new_houses)

# Predictions are already in original scale (not log)
results <- data.frame(
  address = new_houses$address,
  predicted_price = predicted_prices,
  predicted_price_formatted = paste0("$", format(predicted_prices, big.mark = ","))
)

# Save model for production
save_easyNNR_model(
  price_model,
  output_dir = "production/price_model",
  save_plots = TRUE,
  save_best = TRUE
)
```

### Example 3: Multi-Model Comparison

```r
library(easyNNR)

# Load data
data <- read.csv("data.csv")

# Model 1: Simple shallow network
model_shallow <- easy_nn(
  data = data,
  target = "outcome",
  layers = c(64),
  epochs = 50,
  verbose = FALSE
)

# Model 2: Deep network with regularization
model_deep <- easy_nn(
  data = data,
  target = "outcome",
  layers = c(256, 128, 64, 32),
  dropout = 0.3,
  batch_norm = TRUE,
  epochs = 100,
  verbose = FALSE
)

# Model 3: With feature selection
model_selected <- easy_nn(
  data = data,
  target = "outcome",
  layers = c(128, 64),
  preprocess = list(
    feature_selection = "mutual_info",
    n_features = 20
  ),
  epochs = 75,
  verbose = FALSE
)

# Compare models
comparison <- easy_compare(
  model_shallow,
  model_deep,
  model_selected,
  metrics = c("accuracy", "f1_score")
)

print(comparison)

# Visualize comparison
# (assuming comparison returns a data frame)
library(ggplot2)
ggplot(comparison, aes(x = model, y = accuracy, fill = model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Comparison", y = "Accuracy")
```

### Example 4: Time Series Forecasting

```r
library(easyNNR)

# Load time series data
sales <- read.csv("daily_sales.csv")
sales$date <- as.Date(sales$date)

# Sort by date
sales <- sales[order(sales$date), ]

# Time series preprocessing
sales_model <- easy_nn(
  data = sales,
  target = "sales",
  task = "regression",
  
  # Architecture
  layers = c(128, 64, 32),
  dropout = 0.2,
  
  # Time series features
  preprocess = list(
    date_features = TRUE,
    lag_features = c(1, 7, 14, 30, 365),
    rolling_features = list(
      mean_7 = list(window = 7, fun = mean),
      mean_30 = list(window = 30, fun = mean),
      std_7 = list(window = 7, fun = sd),
      max_30 = list(window = 30, fun = max),
      min_30 = list(window = 30, fun = min)
    ),
    target_transform = "log1p",
    outlier_method = "winsorize"
  ),
  
  test_split = 0.1,  # Last 10% for testing
  epochs = 150,
  batch_size = 32,
  seed = 42
)

# Forecast next period
# (Create future data with lagged values)
future_data <- create_future_data(sales, days = 30)
forecasts <- easy_predict(sales_model, future_data)

# Plot forecast
plot_forecast(sales$date, sales$sales, 
              future_data$date, forecasts)
```

### Example 5: Production Deployment Workflow

```r
library(easyNNR)

# ============================================================
# TRAINING PHASE
# ============================================================

# Start logging
start_easyNNR_log("logs/training/")

# Load and prepare data
train_data <- read.csv("training_data.csv")

cat("Training model on", nrow(train_data), "samples\n")

# Train production model
prod_model <- easy_nn(
  data = train_data,
  target = "target_variable",
  
  # Optimal architecture (from tuning)
  layers = c(256, 128, 64),
  activations = "relu",
  dropout = 0.2,
  batch_norm = TRUE,
  
  # Training config
  learning_rate = 0.0005,
  epochs = 150,
  batch_size = 64,
  early_stopping = TRUE,
  patience = 20,
  
  # Production preprocessing
  preprocess = list(
    outlier_method = "winsorize",
    feature_selection = "mutual_info",
    n_features = 50,
    encoding = "target",
    impute_numeric = "median",
    add_indicators = TRUE
  ),
  
  scale_data = "standard",
  seed = 42,
  verbose = TRUE
)

# Save production model
save_easyNNR_model(
  prod_model,
  output_dir = "production/models/v1.0",
  save_plots = TRUE,
  save_logs = TRUE,
  save_best = TRUE
)

# End logging
end_easyNNR_log()

cat("\n✅ Model training complete!\n")
cat("Model saved to: production/models/v1.0\n")

# ============================================================
# INFERENCE PHASE
# ============================================================

# Load production model
inference_model <- load_easyNNR_model(
  "production/models/v1.0/models/model_architecture.rds"
)

cat("\n✅ Model loaded successfully!\n")

# Load new data for scoring
new_data <- read.csv("new_data.csv")

cat("Scoring", nrow(new_data), "samples\n")

# Batch prediction for large dataset
predictions <- easy_predict_batch(
  inference_model,
  new_data,
  batch_size = 1000,
  verbose = TRUE
)

# Get confidence scores
confidence <- easy_predict_confidence(
  inference_model,
  new_data,
  n_iterations = 50,
  verbose = TRUE
)

# Create output
output <- data.frame(
  id = new_data$id,
  prediction = predictions,
  uncertainty = confidence$uncertainty,
  timestamp = Sys.time()
)

# Flag uncertain predictions
output$needs_review <- output$uncertainty > quantile(output$uncertainty, 0.95)

# Save results
write.csv(
  output,
  paste0("production/predictions/predictions_", 
         format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"),
  row.names = FALSE
)

cat("\n✅ Predictions saved!\n")
cat("Flagged", sum(output$needs_review), "samples for review\n")
```

---

## 💾 Model Persistence

### Saving Models

```r
save_easyNNR_model(
  model = trained_model,
  output_dir = "path/to/save/directory",
  save_plots = TRUE,      # Save visualization plots
  save_logs = TRUE,       # Save training summary
  save_best = TRUE        # Save best model checkpoint
)
```

**What gets saved:**
```
output_dir/
├── models/
│   ├── model_weights.h5          # Neural network weights (HDF5)
│   ├── model_architecture.rds    # Model configuration (R object)
│   ├── preprocessing.rds          # Preprocessing pipeline
│   └── best_model_{metric}_{timestamp}/  # Best checkpoint (optional)
├── plots/                         # All visualization plots (if save_plots=TRUE)
│   ├── training_history.png
│   ├── actual_vs_predicted.png   # Regression
│   ├── confusion_matrix.png      # Classification
│   ├── residuals.png
│   └── residual_distribution.png
├── logs/                          # Training logs (if save_logs=TRUE)
│   └── summary_report.txt
├── metrics_{timestamp}.csv
├── training_history_{timestamp}.csv
└── test_predictions_{timestamp}.csv
```

### Loading Models

```r
# Load saved model
loaded_model <- load_easyNNR_model(
  "path/to/save/directory/models/model_architecture.rds"
)

# Model is ready to use immediately
predictions <- easy_predict(loaded_model, new_data)
```

**Complete workflow:**
```r
# Train and save
model <- easy_nn(data = train_data, target = "y")
save_easyNNR_model(model, "results/model_v1")

# Later... load and predict
model <- load_easyNNR_model("results/model_v1/models/model_architecture.rds")
predictions <- easy_predict(model, test_data)
```

---

## 📊 Visualization

### Training History

```r
# Basic plot
easy_plot(model)

# Customized
easy_plot(
  model,
  metrics = c("loss", "accuracy"),
  smooth = TRUE,
  theme = "classic",
  title = "Training Progress"
)
```

### Regression Plots

```r
# Actual vs predicted
easy_plot_regression(model)

# Residual analysis
easy_plot_residuals(model)
easy_plot_residual_dist(model)
```

### Classification Plots

```r
# Confusion matrix
easy_plot_confusion(model, normalize = TRUE)

# Class distribution
easy_plot_class_distribution(model)
```

### Feature Importance

```r
# Top 20 features
easy_plot_importance(model, n_features = 20)
```

### Combining Plots

```r
library(patchwork)

# Create multiple plots
p1 <- easy_plot(model)
p2 <- easy_plot_regression(model)
p3 <- easy_plot_residuals(model)
p4 <- easy_plot_residual_dist(model)

# Combine into grid
(p1 + p2) / (p3 + p4)

# Save combined plot
ggsave("combined_analysis.png", width = 16, height = 12, dpi = 300)
```

---

## 🎯 Prediction Functions

### Standard Prediction

```r
# Classification: get classes
classes <- easy_predict(model, new_data, type = "class")

# Classification: get probabilities
probs <- easy_predict(model, new_data, type = "prob")

# Regression: get values
values <- easy_predict(model, new_data)
```

### Batch Prediction (Large Datasets)

```r
# Memory-efficient prediction
predictions <- easy_predict_batch(
  model,
  large_dataset,
  batch_size = 1000
)
```

### Uncertainty Estimation

```r
# Get predictions with uncertainty
result <- easy_predict_confidence(
  model,
  new_data,
  n_iterations = 100
)

# Access results
predictions <- result$predictions
uncertainty <- result$uncertainty

# Find uncertain predictions
uncertain_idx <- which(uncertainty > quantile(uncertainty, 0.90))
```

---

## ✅ Best Practices

### 1. Data Preparation

```r
# ✅ DO: Clean your data
data <- data[!duplicated(data), ]  # Remove duplicates
data <- data[complete.cases(data[, c("important_column")]), ]  # Handle critical NAs

# ✅ DO: Check data types
str(data)
summary(data)

# ❌ DON'T: Include ID columns
# Remove them with exclude parameter
```

### 2. Model Development

```r
# ✅ DO: Start simple
model_v1 <- easy_nn(
  data = data,
  target = "y",
  layers = c(64, 32),  # Simple architecture
  epochs = 50
)

# ✅ DO: Gradually increase complexity
model_v2 <- easy_nn(
  data = data,
  target = "y",
  layers = c(128, 64, 32),
  dropout = 0.2,
  epochs = 100
)

# ❌ DON'T: Start with overly complex models
```

### 3. Preprocessing

```r
# ✅ DO: Handle class imbalance
model <- easy_nn(
  data = imbalanced_data,
  target = "rare_event",
  preprocess = list(
    imbalance_method = "smote"
  )
)

# ✅ DO: Transform skewed targets
model <- easy_nn(
  data = data,
  target = "price",
  preprocess = list(
    target_transform = "log"
  )
)

# ❌ DON'T: Apply all preprocessing blindly
# Only use what you need
```

### 4. Training

```r
# ✅ DO: Use early stopping
model <- easy_nn(
  data = data,
  target = "y",
  epochs = 200,  # Set high
  early_stopping = TRUE,
  patience = 20  # Let it decide when to stop
)

# ✅ DO: Monitor training curves
easy_plot(model)

# ❌ DON'T: Train for fixed epochs without monitoring
```

### 5. Evaluation

```r
# ✅ DO: Use multiple evaluation plots
easy_plot(model)
easy_plot_regression(model)
easy_plot_residuals(model)
easy_plot_importance(model)

# ✅ DO: Test on truly unseen data
holdout_predictions <- easy_predict(model, holdout_data)

# ❌ DON'T: Only look at training metrics
```

### 6. Production

```r
# ✅ DO: Save everything
save_easyNNR_model(
  model,
  output_dir = "production/model_v1",
  save_plots = TRUE,
  save_logs = TRUE,
  save_best = TRUE
)

# ✅ DO: Version your models
save_easyNNR_model(model, "models/v1.0.0")
save_easyNNR_model(model, "models/v1.1.0")

# ✅ DO: Log everything
start_easyNNR_log("logs/")
# ... training code ...
end_easyNNR_log()

# ❌ DON'T: Deploy without saving preprocessing
```

### 7. Reproducibility

```r
# ✅ DO: Set seeds
model <- easy_nn(
  data = data,
  target = "y",
  seed = 42
)

# ✅ DO: Document preprocessing
preprocess_config <- list(
  outlier_method = "winsorize",
  feature_selection = "mutual_info",
  n_features = 30,
  target_transform = "log"
)

model <- easy_nn(
  data = data,
  target = "y",
  preprocess = preprocess_config,
  seed = 42
)

# Save config for reproducibility
saveRDS(preprocess_config, "config/preprocess.rds")
```

---

## 🔧 Troubleshooting

### Installation Issues

**Problem: Package installation fails**
```r
# Solution: Install dependencies manually
install.packages(c("keras", "tensorflow", "dplyr", "recipes", 
                   "ggplot2", "tidyr", "tibble"))
devtools::install_github("AmeerTamoorKhan/easyNNR")
```

**Problem: TensorFlow not found**
```r
# Solution: Install backend
library(easyNNR)
install_easyNNR_backend()

# Or manually
keras::install_keras(version = "2.15.0", tensorflow = "2.15.0")
```

### Training Issues

**Problem: Model not converging**
```r
# Solution 1: Lower learning rate
model <- easy_nn(data = data, target = "y", learning_rate = 0.0001)

# Solution 2: Increase epochs
model <- easy_nn(data = data, target = "y", epochs = 200)

# Solution 3: Try different optimizer
model <- easy_nn(data = data, target = "y", optimizer = "adamw")

# Solution 4: Add batch normalization
model <- easy_nn(data = data, target = "y", batch_norm = TRUE)
```

**Problem: Overfitting (high train accuracy, low test accuracy)**
```r
# Solution 1: Add dropout
model <- easy_nn(data = data, target = "y", dropout = 0.3)

# Solution 2: Reduce model capacity
model <- easy_nn(data = data, target = "y", layers = c(64, 32))

# Solution 3: More aggressive early stopping
model <- easy_nn(data = data, target = "y", patience = 10)

# Solution 4: Feature selection
model <- easy_nn(
  data = data,
  target = "y",
  preprocess = list(feature_selection = "mutual_info", n_features = 20)
)
```

**Problem: Underfitting (low train and test accuracy)**
```r
# Solution 1: Increase model capacity
model <- easy_nn(data = data, target = "y", layers = c(256, 128, 64))

# Solution 2: Train longer
model <- easy_nn(data = data, target = "y", epochs = 200)

# Solution 3: Feature engineering
model <- easy_nn(
  data = data,
  target = "y",
  preprocess = list(polynomial_degree = 2)
)
```

### Prediction Issues

**Problem: Predictions out of expected range (regression)**
```r
# Check if target transformation was used
print(model$preprocessing$target_transformer)

# Verify inverse transformation is working
pred <- easy_predict(model, test_data)
range(pred)  # Should be in original scale

# If still wrong, manually inverse transform
transformer <- model$preprocessing$target_transformer
if (transformer$method == "log") {
  pred_corrected <- exp(pred)
}
```

**Problem: Poor predictions on new data**
```r
# Solution 1: Check data schema matches
colnames(training_data)
colnames(new_data)

# Solution 2: Check for data drift
summary(training_data)
summary(new_data)

# Solution 3: Ensure preprocessing applies correctly
baked <- recipes::bake(model$recipe, new_data)
head(baked)
```

### Memory Issues

**Problem: Out of memory during training**
```r
# Solution 1: Reduce batch size
model <- easy_nn(data = data, target = "y", batch_size = 16)

# Solution 2: Reduce model size
model <- easy_nn(data = data, target = "y", layers = c(64, 32))

# Solution 3: Use feature selection
model <- easy_nn(
  data = data,
  target = "y",
  preprocess = list(
    feature_selection = "mutual_info",
    n_features = 30
  )
)

# Solution 4: Sample data
data_sample <- data[sample(nrow(data), 10000), ]
model <- easy_nn(data = data_sample, target = "y")
```

**Problem: Out of memory during prediction**
```r
# Solution: Use batch prediction
predictions <- easy_predict_batch(
  model,
  large_data,
  batch_size = 500
)
```

### Performance Issues

**Problem: Training is very slow**
```r
# Check if GPU is being used
tensorflow::tf$config$list_physical_devices("GPU")

# If no GPU:
# - Reduce batch size
# - Reduce model complexity
# - Use fewer epochs
# - Sample data

# Speed up with larger batches (if GPU available)
model <- easy_nn(data = data, target = "y", batch_size = 128)
```

### Data Issues

**Problem: "Target column not found"**
```r
# Check column names
names(data)

# Ensure exact match (case-sensitive)
model <- easy_nn(data = iris, target = "Species")  # ✅
model <- easy_nn(data = iris, target = "species")  # ❌
```

**Problem: Categorical encoding fails**
```r
# Too many categories
# Solution: Use different encoding
model <- easy_nn(
  data = data,
  target = "y",
  preprocess = list(
    encoding = "target",  # Instead of onehot
    max_categories = 100
  )
)
```

**Problem: Missing values cause errors**
```r
# Explicitly configure imputation
model <- easy_nn(
  data = data,
  target = "y",
  preprocess = list(
    impute_numeric = "median",
    impute_categorical = "mode"
  )
)
```

### Getting Help

If you encounter issues not covered here:

1. **Check documentation**: `?easy_nn`
2. **View examples**: Look at successful examples in this README
3. **Enable verbose output**: `verbose = TRUE`
4. **Check logs**: Use logging system
5. **Report issues**: GitHub issues page

```r
# Enable maximum verbosity
start_easyNNR_log("debug_logs/")

model <- easy_nn(
  data = data,
  target = "y",
  verbose = TRUE
)

end_easyNNR_log()

# Review debug_logs/easyNNR_log_*.txt for details
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report bugs**: Open an issue on GitHub
2. **Suggest features**: Open an issue with enhancement tag
3. **Submit PRs**: Fork, branch, code, test, submit
4. **Improve docs**: Fix typos, add examples, clarify explanations

### Development Setup

```r
# Clone repository
git clone https://github.com/AmeerTamoorKhan/easyNNR.git

# Install development dependencies
devtools::install_dev_deps()

# Run tests
devtools::test()

# Check package
devtools::check()

# Build documentation
devtools::document()
```

---

## 📜 License

MIT License - see LICENSE file for details

---

## 📧 Contact

- **GitHub**: https://github.com/AmeerTamoorKhan/easyNNR
- **Issues**: https://github.com/AmeerTamoorKhan/easyNNR/issues

---

## 🙏 Acknowledgments

Built with:
- [TensorFlow](https://www.tensorflow.org/) - Deep learning backend
- [Keras](https://keras.io/) - Neural network API
- [recipes](https://recipes.tidymodels.org/) - Preprocessing framework
- [ggplot2](https://ggplot2.tidyverse.org/) - Visualization
- [tidyverse](https://www.tidyverse.org/) - Data manipulation

---

## 📊 Citation

If you use easyNNR in your research, please cite:

```bibtex
@software{easynr2025,
  title = {easyNNR: Easy Neural Networks in R},
  author = {Khan, Ameer Tamoor},
  year = {2025},
  version = {2.0.0},
  url = {https://github.com/AmeerTamoorKhan/easyNNR}
}
```

---

**Happy modeling with easyNNR! 🚀🧠**
