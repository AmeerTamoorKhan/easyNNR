📘 easyNNR: Easy Neural Networks in R
Simple, automatic, and powerful neural networks for beginners and researchers
🚀 Overview

easyNNR is an easy-to-use R package that allows users to build neural networks for regression and classification with minimal code, while still offering advanced capabilities such as:

Automatic preprocessing

Outlier handling

Feature selection

Training history

Residual analysis

Confidence predictions

Batch prediction

Full model saving & loading

Logging system

Built-in visualization tools

The package is designed for students, researchers, and data scientists who want clean, simple neural network workflows in R without touching TensorFlow/Keras manually.

🛠 Installation
✓ Install from GitHub
install.packages("devtools")
devtools::install_github("AmeerTamoorKhan/easyNNR")

✓ Load the package
library(easyNNR)

⭐ Quick Start Example
Train a classification model on the iris dataset:
model <- easy_nn(
  data = iris,
  target = "Species",
  task = "classification"
)

easy_summary(model)
easy_predict(model, iris[1:5, ])

📊 Regression Example (Wine Dataset)
wine <- read.csv("winequality-red.csv", sep = ";")

model <- easy_nn(
  data = wine,
  target = "quality",
  task = "regression",
  layers = c(128, 64),
  activations = "relu",
  epochs = 60,
  scale_data = "standard"
)

easy_summary(model)
easy_plot(model)

🎯 Key Features
🚦 1. Automatic Preprocessing

One-hot encoding

Scaling

Target transformations

Outlier handling

Missing value cleaning

Mutual information feature selection

🔮 2. Easy Model Training

Regression & Classification

Arbitrary-layer architecture

Dropout & batch normalization

Early stopping

Automatic validation split

📈 3. Built-in Visualization

Training curves

Residual plots

Prediction scatter

Class distributions

Confusion matrix

🔁 4. Prediction Functions

easy_predict() → single prediction

easy_predict_batch() → large datasets

easy_predict_confidence() → uncertainty estimation

💾 5. Full Save/Load System (v2)

Models are saved in 3 parts:

model_architecture.rds

model_weights.h5

preprocessing.rds

Load cleanly across R sessions!

save_easyNNR_model(model, "results/wine_model")
loaded <- load_easyNNR_model("results/wine_model/models/model_architecture.rds")

📝 6. Logging System

Capture all console output:

start_easyNNR_log("logs/")
model <- easy_nn(iris, target = "Species")
end_easyNNR_log()

📦 Saving a Model (Full Example)
save_easyNNR_model(
  model,
  output_dir = "results/wine_red",
  save_plots = TRUE,
  save_logs = TRUE,
  save_best = TRUE
)


This creates:

results/wine_red/
  ├── models/
  │     ├── model_weights.h5
  │     ├── model_architecture.rds
  │     ├── preprocessing.rds
  │     └── best_model_0.87_20251127/
  ├── plots/
  │     ├── training_history.png
  │     ├── actual_vs_predicted.png
  │     └── residuals.png
  ├── logs/
  │     └── summary_report.txt
  ├── metrics_*.csv
  ├── training_history_*.csv
  └── test_predictions_*.csv

🔁 Loading a Saved Model
loaded <- load_easyNNR_model(
  "results/wine_red/models/model_architecture.rds"
)

easy_predict(loaded, wine[1:5, ])

🧪 Structure of an easyNNR Object

A trained model returns:

model$parameters        # architecture, layers, loss, etc.
model$model             # actual Keras model
model$recipe            # preprocessing recipe
model$preprocessing     # settings used
model$evaluation        # all test metrics
model$history           # epoch-by-epoch training history
model$predictions       # saved predictions for test set

🔍 Plotting Tools
easy_plot(model)                  # training curve
easy_plot_regression(model)       # actual vs predicted
easy_plot_residuals(model)        # residuals
easy_plot_residual_dist(model)    # distribution
easy_plot_confusion(model)        # classification only
easy_plot_class_distribution(model)
easy_plot_importance(model)

🔥 Advanced Example (Custom NN)
model <- easy_nn(
  data = iris,
  target = "Species",
  layers = c(256, 128, 64),
  activations = "relu",
  dropout = 0.2,
  batch_norm = TRUE,
  epochs = 100,
  learning_rate = 0.0005,
  scale_data = "minmax"
)

🧰 Functions Summary
Category	Functions
Training	easy_nn()
Prediction	easy_predict(), easy_predict_batch(), easy_predict_confidence()
Visualization	easy_plot(), easy_plot_regression(), easy_plot_confusion(), easy_plot_residuals(), easy_plot_importance()
Saving/Loading	save_easyNNR_model(), load_easyNNR_model()
Logging	start_easyNNR_log(), end_easyNNR_log()
Comparison	easy_compare()
Dashboard	easy_dashboard()
❓ FAQ
Q1: Do I need TensorFlow installed?

Yes, but the package can install it automatically:

easyNNR::install_easyNNR_tensorflow()

Q2: Can I use categorical data?

Yes, one-hot encoding is automatic.

Q3: Can I run this on GPU?

Yes — if TensorFlow detects CUDA.

Q4: Are the models portable?

Yes.
They can be saved and loaded in any R session (v2 system).

🧑‍💻 Contributing

Feel free to submit issues or pull requests:

👉 https://github.com/AmeerTamoorKhan/easyNNR

📄 License

MIT License
