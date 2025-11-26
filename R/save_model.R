# ================================================================
#  easyNNR Save & Load System (v2)
#  Fully reloadable across R sessions
# ================================================================

#' Save easyNNR Model (architecture + weights + preprocessing + results)
#'
#' @description
#' Stores the entire training output in a portable format:
#'   - model architecture (R list)
#'   - model weights (HDF5)
#'   - preprocessing recipe
#'   - metrics, predictions, logs, plots
#'
#' @export
save_easyNNR_model <- function(
    model,
    output_dir,
    save_plots = TRUE,
    save_logs = TRUE,
    save_best = TRUE
) {
  
  if (!inherits(model, "easyNNR"))
    stop("Model must be an easyNNR object")
  
  if (missing(output_dir))
    stop("output_dir is required")
  
  # ----------------------------
  # Directory Structure
  # ----------------------------
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  models_dir <- file.path(output_dir, "models")
  plots_dir  <- file.path(output_dir, "plots")
  logs_dir   <- file.path(output_dir, "logs")
  
  dir.create(models_dir, showWarnings = FALSE)
  if (save_plots) dir.create(plots_dir, showWarnings = FALSE)
  if (save_logs)  dir.create(logs_dir, showWarnings = FALSE)
  
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  saved_files <- list()
  
  cat("\n📦 Saving easyNNR model (portable format)...\n")
  
  # ----------------------------
  # Save model weights
  # ----------------------------
  weights_path <- file.path(models_dir, "model_weights.h5")
  keras::save_model_weights_hdf5(model$model, weights_path)
  saved_files$weights <- weights_path
  cat("  ✓ Saved model weights (HDF5)\n")
  
  # ----------------------------
  # Save model architecture (NOT keras object)
  # ----------------------------
  
  arch <- list(
    layers = model$parameters$layers,
    activations = model$parameters$activations,
    dropout = model$parameters$dropout,
    batch_norm = model$parameters$batch_norm,
    input_dim = model$parameters$input_dim,
    task = model$parameters$task,
    num_classes = model$parameters$num_classes,
    learning_rate = model$parameters$learning_rate,
    optimizer = model$parameters$optimizer,
    loss = model$parameters$loss,
    metrics = model$parameters$metrics
  )
  
  arch_path <- file.path(models_dir, "model_architecture.rds")
  saveRDS(arch, arch_path)
  saved_files$architecture <- arch_path
  cat("  ✓ Saved model architecture\n")
  
  # ----------------------------
  # Save preprocessing objects
  # ----------------------------
  prep <- list(
    recipe = model$recipe,
    preprocessing = model$preprocessing,
    target = model$target,
    class_levels = model$parameters$class_levels
  )
  
  prep_path <- file.path(models_dir, "preprocessing.rds")
  saveRDS(prep, prep_path)
  saved_files$preprocessing <- prep_path
  cat("  ✓ Saved preprocessing pipeline\n")
  
  # ----------------------------
  # Save metrics
  # ----------------------------
  metrics_path <- file.path(output_dir, paste0("metrics_", timestamp, ".csv"))
  write.csv(model$evaluation, metrics_path, row.names = TRUE)
  saved_files$metrics <- metrics_path
  cat("  ✓ Saved evaluation metrics\n")
  
  # ----------------------------
  # Save training history
  # ----------------------------
  if (!is.null(model$history)) {
    hist_path <- file.path(output_dir, paste0("training_history_", timestamp, ".csv"))
    write.csv(model$history, hist_path, row.names = FALSE)
    saved_files$history <- hist_path
    cat("  ✓ Saved training history\n")
  }
  
  # ----------------------------
  # Save predictions table
  # ----------------------------
  pred_path <- file.path(output_dir, paste0("test_predictions_", timestamp, ".csv"))
  write.csv(model$predictions, pred_path, row.names = FALSE)
  saved_files$predictions <- pred_path
  cat("  ✓ Saved test predictions\n")
  
  # ----------------------------
  # Save best model copy
  # ----------------------------
  if (save_best) {
    metric_value <- if (arch$task == "classification") {
      round(model$evaluation$accuracy * 100, 2)
    } else {
      round(model$evaluation$r_squared, 4)
    }
    
    best_name <- paste0("best_model_", metric_value, "_", timestamp)
    best_dir  <- file.path(models_dir, best_name)
    dir.create(best_dir)
    
    file.copy(weights_path, file.path(best_dir, "model_weights.h5"))
    file.copy(arch_path, file.path(best_dir, "model_architecture.rds"))
    file.copy(prep_path, file.path(best_dir, "preprocessing.rds"))
    
    saved_files$best <- best_dir
    cat("  ✓ Saved BEST model folder\n")
  }
  
  # ----------------------------
  # Save plots
  # ----------------------------
  if (save_plots) {
    cat("\n📈 Generating plots...\n")
    
    # Training curve plot
    try({
      p1 <- easy_plot(model)
      g1 <- file.path(plots_dir, paste0("training_history_", timestamp, ".png"))
      ggplot2::ggsave(g1, p1, dpi = 300, width = 10, height = 6)
      saved_files$train_plot <- g1
    })
    
    # Regression-specific plots
    if (arch$task == "regression") {
      try({
        p2 <- easy_plot_regression(model)
        g2 <- file.path(plots_dir, paste0("actual_vs_predicted_", timestamp, ".png"))
        ggplot2::ggsave(g2, p2, dpi = 300, width = 10, height = 6)
        saved_files$reg_plot <- g2
      })
      
      try({
        p3 <- easy_plot_residuals(model)
        g3 <- file.path(plots_dir, paste0("residuals_", timestamp, ".png"))
        ggplot2::ggsave(g3, p3, dpi = 300, width = 10, height = 6)
        saved_files$resid_plot <- g3
      })
    }
  }
  
  # ----------------------------
  # Save summary report
  # ----------------------------
  if (save_logs) {
    log_path <- file.path(logs_dir, paste0("summary_report_", timestamp, ".txt"))
    writeLines(c(
      "easyNNR Saved Model Report",
      "===========================",
      paste("Time:", Sys.time()),
      paste("Target:", arch$target),
      paste("Input Features:", arch$input_dim),
      paste("Architecture:", paste(arch$layers, collapse=" → ")),
      paste("Optimizer:", arch$optimizer),
      paste("Loss:", arch$loss)
    ), log_path)
    
    saved_files$summary <- log_path
    cat("  ✓ Saved summary report\n")
  }
  
  cat("\n✅ All files saved successfully!\n")
  
  invisible(saved_files)
}

# ======================================================================
# LOAD MODEL (FULL RECONSTRUCTION)
# ======================================================================

#' Load a previously saved easyNNR model
#'
#' @export
load_easyNNR_model <- function(path) {
  
  if (!file.exists(path))
    stop("Architecture file not found: ", path)
  
  # Path is architecture.rds (inside models/)
  base_dir <- dirname(path)
  
  cat("📂 Loading easyNNR model from folder:", base_dir, "\n")
  
  arch  <- readRDS(file.path(base_dir, "model_architecture.rds"))
  prep  <- readRDS(file.path(base_dir, "preprocessing.rds"))
  wpath <- file.path(base_dir, "model_weights.h5")
  
  # ----------------------------
  # Rebuild Keras model
  # ----------------------------
  cat("🔧 Rebuilding Keras model architecture...\n")
  
  input <- keras::layer_input(shape = arch$input_dim)
  
  hidden <- input
  for (i in seq_along(arch$layers)) {
    hidden <- hidden |> keras::layer_dense(
      units = arch$layers[i],
      activation = arch$activations[i]
    )
    
    if (arch$batch_norm) {
      hidden <- hidden |> keras::layer_batch_normalization()
    }
    
    if (!is.null(arch$dropout) && arch$dropout[i] > 0) {
      hidden <- hidden |> keras::layer_dropout(rate = arch$dropout[i])
    }
  }
  
  # Output
  if (arch$task == "regression") {
    output <- hidden |> keras::layer_dense(units = 1, activation = "linear")
  } else if (arch$task == "classification" && arch$num_classes == 2) {
    output <- hidden |> keras::layer_dense(units = 1, activation = "sigmoid")
  } else {
    output <- hidden |> keras::layer_dense(units = arch$num_classes, activation = "softmax")
  }
  
  keras_model <- keras::keras_model(inputs = input, outputs = output)
  
  # Compile
  keras_model |> keras::compile(
    optimizer = arch$optimizer,
    loss = arch$loss,
    metrics = arch$metrics
  )
  
  # Load weights
  keras::load_model_weights_hdf5(keras_model, wpath)
  
  cat("  ✓ Weights loaded\n")
  
  # ----------------------------
  # Rebuild easyNNR object
  # ----------------------------
  out <- list(
    model = keras_model,
    recipe = prep$recipe,
    preprocessing = prep$preprocessing,
    target = prep$target,
    parameters = arch
  )
  
  class(out) <- "easyNNR"
  
  cat("✓ Model loaded successfully!\n")
  
  return(out)
}
