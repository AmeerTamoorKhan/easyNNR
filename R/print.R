#' Print Method for easyNNR Objects
#'
#' @param x An easyNNR object
#' @param ... Additional arguments (unused)
#'
#' @export
print.easyNNR <- function(x, ...) {
  cat("\n╔════════════════════════════════════════════════╗\n")
  cat("║         🧠 easyNNR Model Summary              ║\n")
  cat("╚════════════════════════════════════════════════╝\n\n")
  
  # Task and target
  cat("Task:", toupper(x$parameters$task), "\n")
  cat("Target Variable:", x$target, "\n\n")
  
  # Architecture
  cat("Architecture:\n")
  cat("  Input Features:", x$parameters$input_dim, "\n")
  cat("  Hidden Layers:", paste(x$parameters$layers, collapse = " → "), "\n")
  cat("  Activations:", paste(x$parameters$activations, collapse = ", "), "\n")
  
  if (!is.null(x$parameters$dropout)) {
    cat("  Dropout:", paste(x$parameters$dropout, collapse = ", "), "\n")
  }
  
  cat("  Batch Normalization:", x$parameters$batch_norm, "\n\n")
  
  # Preprocessing Summary
  preprocess_opts <- x$preprocessing$options
  if (!is.null(preprocess_opts)) {
    active_preprocess <- c()
    
    if (preprocess_opts$outlier_method != "none") {
      active_preprocess <- c(active_preprocess, 
                             paste0("Outliers: ", preprocess_opts$outlier_method))
    }
    if (preprocess_opts$target_transform != "none") {
      active_preprocess <- c(active_preprocess,
                             paste0("Target: ", preprocess_opts$target_transform))
    }
    if (preprocess_opts$imbalance_method != "none") {
      active_preprocess <- c(active_preprocess,
                             paste0("Imbalance: ", preprocess_opts$imbalance_method))
    }
    if (preprocess_opts$feature_selection != "none") {
      active_preprocess <- c(active_preprocess,
                             paste0("Selection: ", preprocess_opts$feature_selection))
    }
    if (preprocess_opts$encoding != "onehot") {
      active_preprocess <- c(active_preprocess,
                             paste0("Encoding: ", preprocess_opts$encoding))
    }
    if (!is.null(preprocess_opts$pca_components)) {
      active_preprocess <- c(active_preprocess,
                             paste0("PCA: ", preprocess_opts$pca_components, " components"))
    }
    
    if (length(active_preprocess) > 0) {
      cat("Preprocessing:\n")
      for (pp in active_preprocess) {
        cat("  •", pp, "\n")
      }
      cat("\n")
    }
  }
  
  # Training configuration
  cat("Training Configuration:\n")
  cat("  Optimizer:", x$parameters$optimizer, "(lr =", x$parameters$learning_rate, ")\n")
  cat("  Loss Function:", x$parameters$loss, "\n")
  cat("  Epochs:", x$parameters$epochs, "\n")
  cat("  Batch Size:", x$parameters$batch_size, "\n")
  
  if (x$parameters$early_stopping) {
    cat("  Early Stopping: ENABLED (patience =", x$parameters$patience, ")\n")
  }
  
  if (!is.null(x$preprocessing$class_weights)) {
    cat("  Class Weights: ENABLED\n")
  }
  
  cat("\n")
  
  # Performance
  cat("Performance:\n")
  if (x$parameters$task == "classification") {
    cat("  Test Accuracy:", round(x$evaluation$accuracy * 100, 2), "%\n")
    cat("  Test Loss:", round(x$evaluation$loss, 4), "\n")
    if (!is.null(x$evaluation$f1_score)) {
      cat("  F1 Score:", round(x$evaluation$f1_score, 4), "\n")
    }
  } else {
    cat("  R-squared:", round(x$evaluation$r_squared, 4), "\n")
    cat("  RMSE:", round(x$evaluation$rmse, 4), "\n")
    cat("  MAE:", round(x$evaluation$mae, 4), "\n")
  }
  
  cat("\n💡 Use easy_predict(model, new_data) to make predictions\n")
  cat("💡 Use easy_plot(model) to visualize training history\n")
  cat("💡 Use easy_summary(model) for detailed metrics\n\n")
  
  invisible(x)
}


#' Summary Method for easyNNR Objects
#'
#' @param object An easyNNR object
#' @param ... Additional arguments (unused)
#'
#' @export
summary.easyNNR <- function(object, ...) {
  
  cat("\n╔════════════════════════════════════════════════╗\n")
  cat("║      🧠 easyNNR Detailed Model Summary       ║\n")
  cat("╚════════════════════════════════════════════════╝\n\n")
  
  # ============================================================================
  # Model Configuration
  # ============================================================================
  
  cat("═══════════════════════════════════════════════\n")
  cat("                MODEL CONFIGURATION\n")
  cat("═══════════════════════════════════════════════\n\n")
  
  cat("Task Type:", toupper(object$parameters$task), "\n")
  cat("Target Variable:", object$target, "\n")
  
  if (length(object$exclude) > 0) {
    cat("Excluded Columns:", paste(object$exclude, collapse = ", "), "\n")
  }
  
  if (object$parameters$task == "classification") {
    cat("Number of Classes:", object$parameters$num_classes, "\n")
    cat("Class Labels:", paste(object$parameters$class_levels, collapse = ", "), "\n")
  }
  
  cat("\n")
  
  # ============================================================================
  # Preprocessing Configuration
  # ============================================================================
  
  cat("═══════════════════════════════════════════════\n")
  cat("             PREPROCESSING PIPELINE\n")
  cat("═══════════════════════════════════════════════\n\n")
  
  preprocess_opts <- object$preprocessing$options
  
  if (!is.null(preprocess_opts)) {
    cat("Outlier Handling:\n")
    cat("  Method:", preprocess_opts$outlier_method, "\n")
    if (preprocess_opts$outlier_method != "none") {
      cat("  Threshold:", preprocess_opts$outlier_threshold, "\n")
    }
    cat("\n")
    
    if (object$parameters$task == "regression") {
      cat("Target Transformation:", preprocess_opts$target_transform, "\n")
      if (!is.null(object$preprocessing$target_transformer)) {
        trans <- object$preprocessing$target_transformer
        if (!is.null(trans$lambda)) {
          cat("  Lambda:", round(trans$lambda, 4), "\n")
        }
        if (!is.null(trans$offset) && trans$offset != 0) {
          cat("  Offset:", round(trans$offset, 4), "\n")
        }
      }
      cat("\n")
    }
    
    if (object$parameters$task == "classification") {
      cat("Class Imbalance Handling:\n")
      cat("  Method:", preprocess_opts$imbalance_method, "\n")
      if (!is.null(object$preprocessing$class_weights)) {
        cat("  Class Weights:\n")
        for (nm in names(object$preprocessing$class_weights)) {
          cat("    Class", nm, ":", 
              round(object$preprocessing$class_weights[[nm]], 4), "\n")
        }
      }
      cat("\n")
    }
    
    cat("Feature Selection:\n")
    cat("  Method:", preprocess_opts$feature_selection, "\n")
    if (!is.null(object$preprocessing$selected_features)) {
      n_selected <- length(object$preprocessing$selected_features)
      cat("  Features Selected:", n_selected, "\n")
      cat("  Top Features:", 
          paste(head(object$preprocessing$selected_features, 5), collapse = ", "))
      if (n_selected > 5) cat(", ...")
      cat("\n")
    }
    cat("\n")
    
    cat("Encoding Method:", preprocess_opts$encoding, "\n")
    cat("Numeric Imputation:", preprocess_opts$impute_numeric, "\n")
    cat("Categorical Imputation:", preprocess_opts$impute_categorical, "\n")
    cat("Missingness Indicators:", preprocess_opts$add_indicators, "\n\n")
    
    if (preprocess_opts$date_features) {
      cat("Time Series Features: ENABLED\n")
      if (!is.null(preprocess_opts$lag_features)) {
        cat("  Lag Features:", paste(preprocess_opts$lag_features, collapse = ", "), "\n")
      }
      if (!is.null(preprocess_opts$rolling_features)) {
        cat("  Rolling Windows:", 
            paste(preprocess_opts$rolling_features$windows, collapse = ", "), "\n")
      }
      cat("\n")
    }
    
    if (!is.null(preprocess_opts$pca_components)) {
      cat("Dimensionality Reduction:\n")
      cat("  PCA Components:", preprocess_opts$pca_components, "\n\n")
    }
    
    if (isTRUE(preprocess_opts$interactions) || preprocess_opts$polynomial_degree > 1) {
      cat("Feature Engineering:\n")
      if (isTRUE(preprocess_opts$interactions)) {
        cat("  Interaction Terms: ENABLED\n")
      }
      if (preprocess_opts$polynomial_degree > 1) {
        cat("  Polynomial Degree:", preprocess_opts$polynomial_degree, "\n")
      }
      cat("\n")
    }
  }
  
  # ============================================================================
  # Network Architecture
  # ============================================================================
  
  cat("═══════════════════════════════════════════════\n")
  cat("              NETWORK ARCHITECTURE\n")
  cat("═══════════════════════════════════════════════\n\n")
  
  cat("Input Layer:\n")
  cat("  Features:", object$parameters$input_dim, "\n\n")
  
  cat("Hidden Layers:\n")
  for (i in seq_along(object$parameters$layers)) {
    cat("  Layer", i, ":\n")
    cat("    Units:", object$parameters$layers[i], "\n")
    cat("    Activation:", object$parameters$activations[i], "\n")
    
    if (object$parameters$batch_norm) {
      cat("    Batch Normalization: YES\n")
    }
    
    if (!is.null(object$parameters$dropout) && object$parameters$dropout[i] > 0) {
      cat("    Dropout:", object$parameters$dropout[i], "\n")
    }
    
    cat("\n")
  }
  
  cat("Output Layer:\n")
  if (object$parameters$task == "regression") {
    cat("  Units: 1 (continuous output)\n")
    cat("  Activation: linear\n")
  } else {
    if (object$parameters$num_classes == 2) {
      cat("  Units: 1 (binary classification)\n")
      cat("  Activation: sigmoid\n")
    } else {
      cat("  Units:", object$parameters$num_classes, "(multi-class)\n")
      cat("  Activation: softmax\n")
    }
  }
  
  cat("\n")
  
  # ============================================================================
  # Training Configuration
  # ============================================================================
  
  cat("═══════════════════════════════════════════════\n")
  cat("            TRAINING CONFIGURATION\n")
  cat("═══════════════════════════════════════════════\n\n")
  
  cat("Optimizer:", object$parameters$optimizer, "\n")
  cat("Learning Rate:", object$parameters$learning_rate, "\n")
  cat("Loss Function:", object$parameters$loss, "\n")
  cat("Metrics:", paste(object$parameters$metrics, collapse = ", "), "\n")
  cat("Epochs:", object$parameters$epochs, "\n")
  cat("Batch Size:", object$parameters$batch_size, "\n")
  cat("Validation Split:", object$parameters$validation_split, "\n")
  cat("Test Split:", object$parameters$test_split, "\n")
  
  if (object$parameters$early_stopping) {
    cat("Early Stopping: ENABLED\n")
    cat("  Patience:", object$parameters$patience, "epochs\n")
  } else {
    cat("Early Stopping: DISABLED\n")
  }
  
  cat("Feature Scaling:", object$parameters$scale_data, "\n")
  cat("Random Seed:", object$parameters$seed, "\n")
  
  if (object$parameters$rows_removed > 0) {
    cat("Rows Removed (data quality):", object$parameters$rows_removed, "\n")
  }
  
  cat("\n")
  
  # ============================================================================
  # Model Performance
  # ============================================================================
  
  cat("═══════════════════════════════════════════════\n")
  cat("               MODEL PERFORMANCE\n")
  cat("═══════════════════════════════════════════════\n\n")
  
  if (object$parameters$task == "classification") {
    cat("Test Set Metrics:\n")
    cat("  Accuracy:", round(object$evaluation$accuracy * 100, 2), "%\n")
    cat("  Loss:", round(object$evaluation$loss, 4), "\n")
    
    if (!is.null(object$evaluation$precision)) {
      cat("  Precision:", round(object$evaluation$precision * 100, 2), "%\n")
      cat("  Recall:", round(object$evaluation$recall * 100, 2), "%\n")
      cat("  F1 Score:", round(object$evaluation$f1_score, 4), "\n")
    }
    
    cat("\nConfusion Matrix:\n")
    print(object$evaluation$confusion_matrix)
    cat("\n")
    
    # Class-wise accuracy
    conf <- object$evaluation$confusion_matrix
    class_accuracy <- diag(conf) / rowSums(conf)
    cat("Per-Class Accuracy:\n")
    for (i in seq_along(class_accuracy)) {
      cat("  ", names(class_accuracy)[i], ": ", 
          round(class_accuracy[i] * 100, 2), "%\n", sep = "")
    }
    
  } else {
    cat("Test Set Metrics:\n")
    cat("  R-squared:", round(object$evaluation$r_squared, 4), "\n")
    cat("  RMSE:", round(object$evaluation$rmse, 4), "\n")
    cat("  MAE:", round(object$evaluation$mae, 4), "\n")
    cat("  MSE (Loss):", round(object$evaluation$loss, 4), "\n\n")
    
    cat("Prediction Statistics:\n")
    cat("  Min Prediction:", round(min(object$predictions$predicted), 4), "\n")
    cat("  Max Prediction:", round(max(object$predictions$predicted), 4), "\n")
    cat("  Mean Prediction:", round(mean(object$predictions$predicted), 4), "\n")
    cat("  SD Prediction:", round(sd(object$predictions$predicted), 4), "\n\n")
    
    cat("Residual Statistics:\n")
    cat("  Min Residual:", round(min(object$predictions$residual), 4), "\n")
    cat("  Max Residual:", round(max(object$predictions$residual), 4), "\n")
    cat("  Mean Residual:", round(mean(object$predictions$residual), 4), "\n")
    cat("  SD Residual:", round(sd(object$predictions$residual), 4), "\n")
  }
  
  cat("\n")
  
  # ============================================================================
  # Training History Summary
  # ============================================================================
  
  cat("═══════════════════════════════════════════════\n")
  cat("              TRAINING HISTORY\n")
  cat("═══════════════════════════════════════════════\n\n")
  
  # Get final epoch metrics
  hist <- object$history
  final_epoch <- max(hist$epoch)
  final_metrics <- hist[hist$epoch == final_epoch, ]
  
  cat("Final Epoch (", final_epoch, "):\n", sep = "")
  
  for (metric in unique(final_metrics$metric)) {
    metric_data <- final_metrics[final_metrics$metric == metric, ]
    
    train_val <- metric_data$value[metric_data$dataset == "training"]
    val_val <- metric_data$value[metric_data$dataset == "validation"]
    
    if (length(train_val) > 0) {
      cat("  ", metric, " (train): ", round(train_val, 4), "\n", sep = "")
    }
    if (length(val_val) > 0) {
      cat("  ", metric, " (val): ", round(val_val, 4), "\n", sep = "")
    }
  }
  
  cat("\n")
  cat("═══════════════════════════════════════════════\n\n")
  
  invisible(object)
}


#' Get Concise Model Summary as Tibble
#'
#' @description
#' Returns a one-row tibble with key model information.
#' Useful for comparing multiple models or creating summary tables.
#'
#' @param object An easyNNR object
#'
#' @return A tibble with model summary statistics
#'
#' @examples
#' \dontrun{
#' model <- easy_nn(iris, target = "Species")
#' summary_table <- easy_summary(model)
#' print(summary_table)
#' }
#'
#' @export
easy_summary <- function(object) {
  stopifnot(inherits(object, "easyNNR"))
  
  preprocess_opts <- object$preprocessing$options
  
  # Build summary tibble
  summary_tbl <- tibble::tibble(
    task = object$parameters$task,
    target = object$target,
    input_features = object$parameters$input_dim,
    architecture = paste(object$parameters$layers, collapse = "-"),
    activations = paste(object$parameters$activations, collapse = ","),
    optimizer = object$parameters$optimizer,
    learning_rate = object$parameters$learning_rate,
    loss = object$parameters$loss,
    epochs = object$parameters$epochs,
    batch_size = object$parameters$batch_size,
    validation_split = object$parameters$validation_split,
    batch_norm = object$parameters$batch_norm,
    dropout = paste(
      ifelse(is.null(object$parameters$dropout), "-", object$parameters$dropout),
      collapse = ","
    ),
    early_stopping = object$parameters$early_stopping
  )
  
  # Add preprocessing info
  if (!is.null(preprocess_opts)) {
    summary_tbl$outlier_method <- preprocess_opts$outlier_method
    summary_tbl$target_transform <- preprocess_opts$target_transform
    summary_tbl$imbalance_method <- preprocess_opts$imbalance_method
    summary_tbl$feature_selection <- preprocess_opts$feature_selection
    summary_tbl$encoding <- preprocess_opts$encoding
  }
  
  # Add task-specific metrics
  if (object$parameters$task == "classification") {
    summary_tbl$test_accuracy <- object$evaluation$accuracy
    summary_tbl$test_loss <- object$evaluation$loss
    summary_tbl$f1_score <- object$evaluation$f1_score
    summary_tbl$num_classes <- object$parameters$num_classes
  } else {
    summary_tbl$r_squared <- object$evaluation$r_squared
    summary_tbl$rmse <- object$evaluation$rmse
    summary_tbl$mae <- object$evaluation$mae
    summary_tbl$test_loss <- object$evaluation$loss
  }
  
  summary_tbl
}


#' Compare Multiple easyNNR Models
#'
#' @description
#' Create a comparison table of multiple trained models.
#'
#' @param ... easyNNR model objects to compare
#' @param names Optional character vector of model names
#'
#' @return A tibble comparing model configurations and performance
#'
#' @examples
#' \dontrun{
#' model1 <- easy_nn(iris, target = "Species", layers = c(64, 32))
#' model2 <- easy_nn(iris, target = "Species", layers = c(128, 64, 32))
#' comparison <- easy_compare(model1, model2, names = c("Small", "Large"))
#' }
#'
#' @export
easy_compare <- function(..., names = NULL) {
  models <- list(...)
  
  if (length(models) == 0) {
    stop("At least one model must be provided")
  }
  
  # Validate all inputs are easyNNR objects
  for (i in seq_along(models)) {
    if (!inherits(models[[i]], "easyNNR")) {
      stop("All arguments must be easyNNR model objects")
    }
  }
  
  # Generate names if not provided
  if (is.null(names)) {
    names <- paste0("Model_", seq_along(models))
  }
  
  # Get summaries for all models
  summaries <- lapply(models, easy_summary)
  
  # Combine into single tibble
  comparison <- dplyr::bind_rows(summaries)
  comparison$model_name <- names
  
  # Reorder columns to put model_name first
  comparison <- comparison[, c("model_name", setdiff(names(comparison), "model_name"))]
  
  comparison
}