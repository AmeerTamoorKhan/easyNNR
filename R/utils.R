#' Check TensorFlow/Keras Environment
#' @keywords internal
.check_tf <- function() {
  ok <- TRUE
  msg <- NULL
  
  if (!requireNamespace("tensorflow", quietly = TRUE) ||
      !requireNamespace("keras", quietly = TRUE)) {
    ok <- FALSE
    msg <- paste(
      "❌ Packages 'tensorflow' and 'keras' are required.",
      "\n   Install via: install.packages(c('tensorflow', 'keras'))",
      "\n   Then run: easyNNR::install_easyNNR_backend()"
    )
    return(list(ok = ok, msg = msg))
  }
  
  # Load packages quietly
  suppressMessages({
    library(tensorflow)
    library(keras)
  })
  
  # Check if TensorFlow backend is available
  tf_available <- tryCatch({
    tf$constant(1)
    TRUE
  }, error = function(e) FALSE)
  
  if (!tf_available) {
    ok <- FALSE
    msg <- paste(
      "❌ TensorFlow backend not found or not working.",
      "\n   Install compatible backend via:",
      "\n   easyNNR::install_easyNNR_backend()"
    )
  }
  
  list(ok = ok, msg = msg)
}


#' Install Compatible TensorFlow/Keras Backend for easyNNR
#'
#' Ensures TensorFlow 2.15.0 and Keras 2.15.0 are installed —
#' the versions officially compatible with easyNNR.
#' This creates or updates the 'r-tensorflow' virtual environment.
#'
#' @param method Installation method: "auto", "virtualenv", or "conda" (default: "auto")
#' @param ... Additional arguments passed to keras::install_keras()
#'
#' @examples
#' \dontrun{
#' # Install compatible backend
#' install_easyNNR_backend()
#' 
#' # Or specify method
#' install_easyNNR_backend(method = "conda")
#' }
#'
#' @export
install_easyNNR_backend <- function(method = "auto", ...) {
  if (!requireNamespace("keras", quietly = TRUE)) {
    stop("Please install.packages('keras') first.", call. = FALSE)
  }
  
  cat("\n╔════════════════════════════════════════════════╗\n")
  cat("║    🔧 easyNNR Backend Installation           ║\n")
  cat("╚════════════════════════════════════════════════╝\n\n")
  cat("Installing TensorFlow 2.15.0 + Keras 2.15.0...\n")
  cat("This may take a few minutes...\n\n")
  
  tryCatch({
    keras::install_keras(
      version = "2.15.0",
      tensorflow = "2.15.0",
      method = method,
      ...
    )
    
    cat("\n╔════════════════════════════════════════════════╗\n")
    cat("║         ✅ Installation Complete!            ║\n")
    cat("╚════════════════════════════════════════════════╝\n\n")
    cat("easyNNR backend is ready to use!\n")
    cat("Test with: library(easyNNR); easy_nn(iris, 'Species')\n\n")
    
    invisible(TRUE)
    
  }, error = function(e) {
    cat("\n❌ Installation failed!\n")
    cat("Error:", conditionMessage(e), "\n\n")
    cat("Troubleshooting:\n")
    cat("  1. Try different method: install_easyNNR_backend(method='conda')\n")
    cat("  2. Check Python installation: reticulate::py_config()\n")
    cat("  3. See documentation: ?install_easyNNR_backend\n\n")
    
    stop("Backend installation failed. See message above.", call. = FALSE)
  })
}


#' Quick TensorFlow Installation (Legacy Helper)
#'
#' Simple wrapper around keras::install_keras() for backward compatibility.
#' For best results, use install_easyNNR_backend() instead.
#'
#' @param ... Arguments passed to keras::install_keras()
#' @export
install_easyNNR_tensorflow <- function(...) {
  if (!requireNamespace("keras", quietly = TRUE)) {
    stop("Please install.packages('keras') first.", call. = FALSE)
  }
  
  message("🔧 Installing TensorFlow...")
  message("💡 Tip: Use install_easyNNR_backend() for version-controlled installation")
  
  keras::install_keras(...)
  invisible(TRUE)
}


#' Make Keras Optimizer from String or Object
#' @param opt Character string or keras optimizer object
#' @param lr Learning rate (default: 0.001)
#' @keywords internal
.make_optimizer <- function(opt, lr = 0.001) {
  if (is.character(opt)) {
    switch(tolower(opt),
      "adam"     = keras::optimizer_adam(learning_rate = lr),
      "adamw"    = keras::optimizer_adamw(learning_rate = lr),
      "rmsprop"  = keras::optimizer_rmsprop(learning_rate = lr),
      "sgd"      = keras::optimizer_sgd(learning_rate = lr),
      "adagrad"  = keras::optimizer_adagrad(learning_rate = lr),
      "adadelta" = keras::optimizer_adadelta(learning_rate = lr),
      "nadam"    = keras::optimizer_nadam(learning_rate = lr),
      {
        warning("Unknown optimizer '", opt, "'. Falling back to 'adam'.")
        keras::optimizer_adam(learning_rate = lr)
      }
    )
  } else {
    opt
  }
}


#' Infer Task Type from Target Variable
#' @param y Target variable vector
#' @keywords internal
.infer_task <- function(y) {
  if (is.factor(y) || is.character(y)) {
    return("classification")
  }
  
  if (is.numeric(y)) {
    unique_values <- length(unique(y))
    
    # If integer-like and few unique values, likely classification
    if (all(y == floor(y), na.rm = TRUE) && unique_values <= 20) {
      return("classification")
    }
    
    # If many unique values, likely regression
    if (unique_values > 10) {
      return("regression")
    }
    
    # Ambiguous case - default to classification for discrete values
    return("classification")
  }
  
  # Default fallback
  return("regression")
}


#' Convert Keras Training History to Tidy Tibble
#' @param hist Keras history object from fit()
#' @return Tibble with columns: epoch, metric, value
#' @keywords internal
.tidy_history <- function(hist) {
  df <- as.data.frame(hist)
  df$epoch <- seq_len(nrow(df))
  
  # Ensure all columns except epoch are numeric
  df[] <- lapply(df, function(x) {
    if (is.factor(x)) as.numeric(as.character(x)) else x
  })
  
  # Convert to long format
  tidy_df <- tibble::as_tibble(df) |>
    tidyr::pivot_longer(
      cols = -epoch,
      names_to = "metric",
      values_to = "value",
      values_transform = list(value = as.numeric)
    )
  
  # Add training/validation split indicator
  tidy_df$dataset <- ifelse(
    grepl("^val_", tidy_df$metric),
    "validation",
    "training"
  )
  
  # Clean metric names
  tidy_df$metric <- gsub("^val_", "", tidy_df$metric)
  
  tidy_df
}


#' Calculate Classification Metrics
#' @param actual Actual labels (factor or character)
#' @param predicted Predicted labels (factor or character)
#' @return List with accuracy, precision, recall, f1_score, confusion_matrix
#' @keywords internal
.classification_metrics <- function(actual, predicted) {
  confusion <- table(Actual = actual, Predicted = predicted)
  accuracy <- mean(actual == predicted)
  
  # Per-class metrics (if binary or small number of classes)
  classes <- unique(c(as.character(actual), as.character(predicted)))
  
  if (length(classes) == 2) {
    # Binary classification metrics
    pos_class <- classes[2]  # Assume second class is "positive"
    
    tp <- sum(actual == pos_class & predicted == pos_class)
    tn <- sum(actual != pos_class & predicted != pos_class)
    fp <- sum(actual != pos_class & predicted == pos_class)
    fn <- sum(actual == pos_class & predicted != pos_class)
    
    precision <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    recall <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    f1 <- if ((precision + recall) > 0) 2 * precision * recall / (precision + recall) else 0
    
    list(
      confusion_matrix = confusion,
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      f1_score = f1,
      true_positives = tp,
      true_negatives = tn,
      false_positives = fp,
      false_negatives = fn
    )
  } else {
    # Multi-class: calculate macro-averaged metrics
    precision_per_class <- numeric(length(classes))
    recall_per_class <- numeric(length(classes))
    
    for (i in seq_along(classes)) {
      cl <- classes[i]
      tp <- sum(actual == cl & predicted == cl)
      fp <- sum(actual != cl & predicted == cl)
      fn <- sum(actual == cl & predicted != cl)
      
      precision_per_class[i] <- if ((tp + fp) > 0) tp / (tp + fp) else 0
      recall_per_class[i] <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    }
    
    macro_precision <- mean(precision_per_class)
    macro_recall <- mean(recall_per_class)
    macro_f1 <- if ((macro_precision + macro_recall) > 0) {
      2 * macro_precision * macro_recall / (macro_precision + macro_recall)
    } else 0
    
    list(
      confusion_matrix = confusion,
      accuracy = accuracy,
      precision = macro_precision,
      recall = macro_recall,
      f1_score = macro_f1,
      per_class_precision = setNames(precision_per_class, classes),
      per_class_recall = setNames(recall_per_class, classes)
    )
  }
}


#' Calculate Regression Metrics
#' @param actual Actual values
#' @param predicted Predicted values
#' @return List with mae, mse, rmse, r_squared
#' @keywords internal
.regression_metrics <- function(actual, predicted) {
  residuals <- actual - predicted
  
  mae <- mean(abs(residuals))
  mse <- mean(residuals^2)
  rmse <- sqrt(mse)
  
  ss_total <- sum((actual - mean(actual))^2)
  ss_residual <- sum(residuals^2)
  r_squared <- 1 - (ss_residual / ss_total)
  
  # Mean Absolute Percentage Error (MAPE)
  mape <- if (all(actual != 0)) {
    mean(abs(residuals / actual)) * 100
  } else NA
  
  list(
    mae = mae,
    mse = mse,
    rmse = rmse,
    r_squared = r_squared,
    mape = mape,
    mean_actual = mean(actual),
    sd_actual = sd(actual),
    mean_predicted = mean(predicted),
    sd_predicted = sd(predicted)
  )
}


#' Format Number for Printing
#' @param x Numeric value
#' @param digits Number of decimal places (default: 4)
#' @keywords internal
.fmt <- function(x, digits = 4) {
  if (is.null(x) || is.na(x)) return("NA")
  if (!is.numeric(x)) return(as.character(x))
  format(round(x, digits), nsmall = digits)
}


#' Check Package Version
#' @param pkg Package name
#' @param min_version Minimum required version
#' @keywords internal
.check_package_version <- function(pkg, min_version) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    return(FALSE)
  }
  
  current <- packageVersion(pkg)
  required <- package_version(min_version)
  
  current >= required
}


#' Compute VIF (Variance Inflation Factor) for Multicollinearity Detection
#' @param X Feature matrix
#' @return Named vector of VIF values
#' @keywords internal
.compute_vif <- function(X) {
  
  p <- ncol(X)
  if (p < 2) return(setNames(1, colnames(X)[1]))
  
  vif_values <- numeric(p)
  names(vif_values) <- colnames(X)
  
  for (j in seq_len(p)) {
    y_j <- X[, j]
    X_other <- X[, -j, drop = FALSE]
    
    # Fit linear regression: X_j ~ other X's
    fit <- tryCatch({
      lm(y_j ~ X_other)
    }, error = function(e) NULL)
    
    if (is.null(fit)) {
      vif_values[j] <- NA
    } else {
      r_squared <- summary(fit)$r.squared
      vif_values[j] <- if (r_squared < 1) 1 / (1 - r_squared) else Inf
    }
  }
  
  vif_values
}


#' Check for High Multicollinearity
#' @param X Feature matrix
#' @param threshold VIF threshold (default: 10)
#' @return List with high VIF features and recommendations
#' @keywords internal
.check_multicollinearity <- function(X, threshold = 10) {
  
  vif_values <- .compute_vif(X)
  high_vif <- vif_values[vif_values > threshold]
  
  list(
    vif = vif_values,
    high_vif_features = names(high_vif),
    has_multicollinearity = length(high_vif) > 0,
    recommendation = if (length(high_vif) > 0) {
      paste0("Consider removing or combining features: ", 
             paste(names(high_vif), collapse = ", "))
    } else "No multicollinearity detected"
  )
}


#' Calculate Feature Importance from Neural Network
#' @param model Keras model
#' @param X_test Test feature matrix
#' @param y_test Test target vector
#' @param feature_names Character vector of feature names
#' @return Data frame with feature importance scores
#' @keywords internal
.permutation_importance <- function(model, X_test, y_test, feature_names = NULL) {
  
  if (is.null(feature_names)) {
    feature_names <- paste0("V", seq_len(ncol(X_test)))
  }
  
  # Baseline score
  baseline_pred <- model(X_test, training = FALSE) |> as.matrix()
  baseline_score <- mean((baseline_pred - y_test)^2)
  
  importance <- numeric(ncol(X_test))
  names(importance) <- feature_names
  
  for (j in seq_len(ncol(X_test))) {
    # Permute feature j
    X_permuted <- X_test
    X_permuted[, j] <- sample(X_permuted[, j])
    
    # Score with permuted feature
    permuted_pred <- model(X_permuted, training = FALSE) |> as.matrix()
    permuted_score <- mean((permuted_pred - y_test)^2)
    
    # Importance = increase in error when feature is permuted
    importance[j] <- permuted_score - baseline_score
  }
  
  # Normalize
  importance <- importance / sum(abs(importance)) * 100
  
  data.frame(
    feature = feature_names,
    importance = importance,
    rank = rank(-importance)
  ) |> dplyr::arrange(rank)
}


#' Cross-Validation for easyNNR
#' @param data Data frame
#' @param target Target column name
#' @param k Number of folds
#' @param ... Additional arguments passed to easy_nn
#' @return List with CV results
#' @keywords internal
.cross_validate <- function(data, target, k = 5, ...) {
  
  n <- nrow(data)
  fold_size <- floor(n / k)
  folds <- sample(rep(1:k, length.out = n))
  
  results <- vector("list", k)
  
  for (i in seq_len(k)) {
    # Split data
    test_idx <- which(folds == i)
    train_data <- data[-test_idx, , drop = FALSE]
    test_data <- data[test_idx, , drop = FALSE]
    
    # Train model (suppress output)
    model <- easy_nn(
      data = train_data,
      target = target,
      verbose = FALSE,
      ...
    )
    
    # Evaluate on test fold
    predictions <- easy_predict(model, test_data, verbose = FALSE)
    actual <- test_data[[target]]
    
    if (model$parameters$task == "classification") {
      results[[i]] <- list(
        fold = i,
        accuracy = mean(predictions == actual)
      )
    } else {
      results[[i]] <- list(
        fold = i,
        rmse = sqrt(mean((predictions - actual)^2)),
        mae = mean(abs(predictions - actual))
      )
    }
  }
  
  # Aggregate results
  if (model$parameters$task == "classification") {
    accuracies <- sapply(results, `[[`, "accuracy")
    list(
      folds = results,
      mean_accuracy = mean(accuracies),
      sd_accuracy = sd(accuracies)
    )
  } else {
    rmses <- sapply(results, `[[`, "rmse")
    maes <- sapply(results, `[[`, "mae")
    list(
      folds = results,
      mean_rmse = mean(rmses),
      sd_rmse = sd(rmses),
      mean_mae = mean(maes),
      sd_mae = sd(maes)
    )
  }
}