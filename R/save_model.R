# ================================================================
#  easyNNR Save/Load + Logging System (Unified in save_model.R)
# ================================================================

# =====================================================================
# Internal logging environment
# =====================================================================

.easyNNR_log_env <- new.env(parent = emptyenv())


# =====================================================================
# LOGGING SYSTEM
# =====================================================================

#' Start easyNNR Logging
#'
#' @description
#' Starts logging all console output to a timestamped file.
#'
#' @param log_dir Directory where log files should be saved.
#'
#' @export
start_easyNNR_log <- function(log_dir) {
  
  dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
  
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  log_file <- file.path(log_dir, paste0("easyNNR_log_", timestamp, ".txt"))
  
  con <- file(log_file, open = "wt")
  sink(con, type = "output", split = TRUE)
  sink(con, type = "message")
  
  .easyNNR_log_env$connection <- con
  .easyNNR_log_env$path <- log_file
  
  cat("═════════════════════════════════════════\n")
  cat(" easyNNR Logging Started\n")
  cat("═════════════════════════════════════════\n")
  cat("Time: ", Sys.time(), "\n")
  cat("Log file: ", log_file, "\n\n")
  
  invisible(log_file)
}

#' End easyNNR Logging
#'
#' @description
#' Stops logging and closes the log file.
#'
#' @export
end_easyNNR_log <- function() {
  
  if (!exists("connection", envir = .easyNNR_log_env)) {
    message("No active easyNNR logging session found.")
    return(invisible(NULL))
  }
  
  cat("\n═════════════════════════════════════════\n")
  cat(" easyNNR Logging Ended\n")
  cat("Time: ", Sys.time(), "\n")
  cat("═════════════════════════════════════════\n")
  
  sink(type = "output")
  sink(type = "message")
  close(.easyNNR_log_env$connection)
  
  message("✓ Log saved at: ", .easyNNR_log_env$path)
  
  rm(list = c("connection", "path"), envir = .easyNNR_log_env)
  
  invisible(TRUE)
}



# =====================================================================
# SAVE & LOAD SYSTEM
# =====================================================================

#' Save easyNNR Model and Results (architecture + weights + preprocessing)
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
  
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  models_dir <- file.path(output_dir, "models")
  plots_dir  <- file.path(output_dir, "plots")
  logs_dir   <- file.path(output_dir, "logs")
  
  dir.create(models_dir, showWarnings = FALSE)
  if (save_plots) dir.create(plots_dir, showWarnings = FALSE)
  if (save_logs)  dir.create(logs_dir, showWarnings = FALSE)
  
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  saved_files <- list()
  
  cat("\n📦 Saving easyNNR model...\n")
  
  # -----------------------------------------------------
  # Save model weights
  # -----------------------------------------------------
  weights_path <- file.path(models_dir, "model_weights.h5")
  keras::save_model_weights_hdf5(model$model, weights_path)
  saved_files$weights <- weights_path
  cat("  ✓ Model weights saved\n")
  
  # -----------------------------------------------------
  # Save architecture
  # -----------------------------------------------------
  arch <- model$parameters
  arch_path <- file.path(models_dir, "model_architecture.rds")
  saveRDS(arch, arch_path)
  saved_files$architecture <- arch_path
  cat("  ✓ Architecture saved\n")
  
  # -----------------------------------------------------
  # Save preprocessing
  # -----------------------------------------------------
  prep <- list(
    recipe = model$recipe,
    preprocessing = model$preprocessing,
    target = model$target,
    class_levels = model$parameters$class_levels
  )
  
  prep_path <- file.path(models_dir, "preprocessing.rds")
  saveRDS(prep, prep_path)
  saved_files$preprocessing <- prep_path
  cat("  ✓ Preprocessing saved\n")
  
  # -----------------------------------------------------
  # Save metrics
  # -----------------------------------------------------
  metrics_path <- file.path(output_dir, paste0("metrics_", timestamp, ".csv"))
  write.csv(model$evaluation, metrics_path, row.names = TRUE)
  saved_files$metrics <- metrics_path
  cat("  ✓ Metrics saved\n")
  
  # -----------------------------------------------------
  # Save history
  # -----------------------------------------------------
  if (!is.null(model$history)) {
    hist_path <- file.path(output_dir, paste0("training_history_", timestamp, ".csv"))
    write.csv(model$history, hist_path, row.names = FALSE)
    saved_files$history <- hist_path
    cat("  ✓ Training history saved\n")
  }
  
  # -----------------------------------------------------
  # Save predictions
  # -----------------------------------------------------
  pred_path <- file.path(output_dir, paste0("test_predictions_", timestamp, ".csv"))
  write.csv(model$predictions, pred_path, row.names = FALSE)
  saved_files$predictions <- pred_path
  cat("  ✓ Predictions saved\n")
  
  # -----------------------------------------------------
  # Save BEST model
  # -----------------------------------------------------
  if (save_best) {
    score <- if (arch$task == "classification")
      round(model$evaluation$accuracy * 100, 2)
    else
      round(model$evaluation$r_squared, 4)
    
    best_folder <- file.path(models_dir, paste0("best_model_", score, "_", timestamp))
    dir.create(best_folder)
    
    file.copy(weights_path, file.path(best_folder, "weights.h5"))
    file.copy(arch_path, file.path(best_folder, "architecture.rds"))
    file.copy(prep_path, file.path(best_folder, "preprocessing.rds"))
    
    saved_files$best <- best_folder
    cat("  ✓ Best model saved\n")
  }
  
  # -----------------------------------------------------
  # Save plots
  # -----------------------------------------------------
  if (save_plots) {
    cat("\n📈 Saving plots...\n")
    
    try({
      p1 <- easy_plot(model)
      g1 <- file.path(plots_dir, paste0("training_history_", timestamp, ".png"))
      ggplot2::ggsave(g1, p1, dpi = 300, width = 10, height = 6)
      saved_files$training_plot <- g1
    })
    
    if (arch$task == "regression") {
      try({
        p2 <- easy_plot_regression(model)
        g2 <- file.path(plots_dir, paste0("actual_vs_predicted_", timestamp, ".png"))
        ggplot2::ggsave(g2, p2, dpi = 300, width = 10, height = 6)
        saved_files$regression_plot <- g2
      })
    }
  }
  
  # -----------------------------------------------------
  # Summary report
  # -----------------------------------------------------
  if (save_logs) {
    summary_path <- file.path(logs_dir, paste0("summary_report_", timestamp, ".txt"))
    
    writeLines(c(
      "easyNNR Model Summary Report",
      "============================",
      paste("Created:", Sys.time()),
      paste("Task:", arch$task),
      paste("Target:", prep$target),
      paste("Input Dim:", arch$input_dim),
      paste("Layers:", paste(arch$layers, collapse=" → "))
    ), summary_path)
    
    saved_files$summary <- summary_path
    cat("  ✓ Summary report saved\n")
  }
  
  cat("\n✅ All model files saved!\n")
  
  invisible(saved_files)
}


# =====================================================================
# LOAD MODEL
# =====================================================================

#' Load a previously saved easyNNR model
#'
#' @export
load_easyNNR_model <- function(path) {
  
  base_dir <- dirname(path)
  
  arch  <- readRDS(file.path(base_dir, "model_architecture.rds"))
  prep  <- readRDS(file.path(base_dir, "preprocessing.rds"))
  wpath <- file.path(base_dir, "model_weights.h5")
  
  # ----------- Rebuild model ------------
  input <- keras::layer_input(shape = arch$input_dim)
  hidden <- input
  
  for (i in seq_along(arch$layers)) {
    hidden <- hidden |>
      keras::layer_dense(units = arch$layers[i], activation = arch$activations[i])
    
    if (arch$batch_norm)
      hidden <- hidden |> keras::layer_batch_normalization()
    
    if (!is.null(arch$dropout) && arch$dropout[i] > 0)
      hidden <- hidden |> keras::layer_dropout(rate = arch$dropout[i])
  }
  
  output <- if (arch$task == "regression") {
    hidden |> keras::layer_dense(units = 1)
  } else {
    hidden |> keras::layer_dense(units = arch$num_classes, activation = "softmax")
  }
  
  keras_model <- keras::keras_model(inputs = input, outputs = output)
  
  keras_model |> keras::compile(
    optimizer = arch$optimizer,
    loss = arch$loss,
    metrics = arch$metrics
  )
  
  keras::load_model_weights_hdf5(keras_model, wpath)
  
  out <- list(
    model = keras_model,
    recipe = prep$recipe,
    preprocessing = prep$preprocessing,
    target = prep$target,
    parameters = arch
  )
  
  class(out) <- "easyNNR"
  
  message("✓ easyNNR model loaded successfully.")
  return(out)
}
