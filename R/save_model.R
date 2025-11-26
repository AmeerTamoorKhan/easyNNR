#' Save easyNNR Model with Results
#'
#' @description
#' Save a trained easyNNR model along with all results, plots, and logs
#' to a structured output directory.
#'
#' @param model An easyNNR model object
#' @param output_dir Output directory path (will be created if doesn't exist)
#' @param save_plots Logical, whether to save plots (default: TRUE)
#' @param save_logs Logical, whether to save training logs (default: TRUE)
#' @param save_best Logical, whether to save as "best" model with metrics in filename (default: TRUE)
#'
#' @return Invisible list with paths to saved files
#' @export
#'
#' @examples
#' \dontrun{
#' model <- easy_nn(iris, target = "Species")
#' save_easyNNR_model(model, output_dir = "~/my_models")
#' }
save_easyNNR_model <- function(model, 
                               output_dir,
                               save_plots = TRUE,
                               save_logs = TRUE,
                               save_best = TRUE) {
  
  # Validate input
  if (!inherits(model, "easyNNR")) {
    stop("Model must be an easyNNR object")
  }
  
  if (missing(output_dir)) {
    stop("output_dir is required")
  }
  
  # Create directory structure
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  models_dir <- file.path(output_dir, "models")
  plots_dir <- file.path(output_dir, "plots")
  logs_dir <- file.path(output_dir, "logs")
  
  dir.create(models_dir, showWarnings = FALSE)
  if (save_plots) dir.create(plots_dir, showWarnings = FALSE)
  if (save_logs) dir.create(logs_dir, showWarnings = FALSE)
  
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  saved_files <- list()
  
  cat("\n📦 Saving easyNNR model and results...\n\n")
  
  # ============================================================================
  # SAVE MODELS
  # ============================================================================
  
  cat("💾 Saving models...\n")
  
  # Save latest model
  latest_path <- file.path(models_dir, "latest_model.rds")
  saveRDS(model, latest_path)
  saved_files$latest_model <- latest_path
  cat("  ✓ Latest model:", latest_path, "\n")
  
  # Save best model with metrics in filename
  if (save_best) {
    if (model$parameters$task == "classification") {
      metric_value <- round(model$evaluation$accuracy * 100, 2)
      metric_name <- "acc"
    } else {
      metric_value <- round(model$evaluation$r_squared, 4)
      metric_name <- "R2"
    }
    
    best_filename <- paste0("best_model_", metric_name, "_", metric_value, "_", timestamp, ".rds")
    best_path <- file.path(models_dir, best_filename)
    saveRDS(model, best_path)
    saved_files$best_model <- best_path
    cat("  ✓ Best model:", best_path, "\n")
  }
  
  # ============================================================================
  # SAVE RESULTS
  # ============================================================================
  
  cat("\n📊 Saving results...\n")
  
  # Metrics
  if (model$parameters$task == "classification") {
    metrics <- data.frame(
      Metric = c("Accuracy", "Precision", "Recall", "F1-Score", "Loss"),
      Value = c(
        model$evaluation$accuracy,
        model$evaluation$precision,
        model$evaluation$recall,
        model$evaluation$f1_score,
        model$evaluation$loss
      )
    )
  } else {
    metrics <- data.frame(
      Metric = c("R-squared", "RMSE", "MAE", "Loss"),
      Value = c(
        model$evaluation$r_squared,
        model$evaluation$rmse,
        model$evaluation$mae,
        model$evaluation$loss
      )
    )
  }
  
  metrics_path <- file.path(output_dir, paste0("metrics_", timestamp, ".csv"))
  write.csv(metrics, metrics_path, row.names = FALSE)
  saved_files$metrics <- metrics_path
  cat("  ✓ Metrics:", metrics_path, "\n")
  
  # Training history
  if (!is.null(model$history)) {
    history_path <- file.path(output_dir, paste0("training_history_", timestamp, ".csv"))
    write.csv(model$history, history_path, row.names = FALSE)
    saved_files$history <- history_path
    cat("  ✓ Training history:", history_path, "\n")
  }
  
  # Predictions
  if (!is.null(model$predictions)) {
    predictions_path <- file.path(output_dir, paste0("test_predictions_", timestamp, ".csv"))
    write.csv(model$predictions, predictions_path, row.names = FALSE)
    saved_files$predictions <- predictions_path
    cat("  ✓ Predictions:", predictions_path, "\n")
  }
  
  # ============================================================================
  # SAVE PLOTS
  # ============================================================================
  
  if (save_plots) {
    cat("\n📈 Saving plots...\n")
    
    # Training history plot
    tryCatch({
      p1 <- easy_plot(model)
      history_plot_path <- file.path(plots_dir, paste0("training_history_", timestamp, ".png"))
      ggplot2::ggsave(history_plot_path, p1, width = 10, height = 6, dpi = 300)
      saved_files$plot_history <- history_plot_path
      cat("  ✓ Training history plot\n")
    }, error = function(e) {
      cat("  ⚠ Could not save training history plot:", e$message, "\n")
    })
    
    # Task-specific plots
    if (model$parameters$task == "classification") {
      # Confusion matrix
      tryCatch({
        p2 <- easy_plot_confusion(model)
        conf_plot_path <- file.path(plots_dir, paste0("confusion_matrix_", timestamp, ".png"))
        ggplot2::ggsave(conf_plot_path, p2, width = 8, height = 6, dpi = 300)
        saved_files$plot_confusion <- conf_plot_path
        cat("  ✓ Confusion matrix plot\n")
      }, error = function(e) {
        cat("  ⚠ Could not save confusion matrix:", e$message, "\n")
      })
      
    } else {
      # Regression plots
      tryCatch({
        p2 <- easy_plot_regression(model)
        reg_plot_path <- file.path(plots_dir, paste0("actual_vs_predicted_", timestamp, ".png"))
        ggplot2::ggsave(reg_plot_path, p2, width = 8, height = 6, dpi = 300)
        saved_files$plot_regression <- reg_plot_path
        cat("  ✓ Actual vs predicted plot\n")
      }, error = function(e) {
        cat("  ⚠ Could not save regression plot:", e$message, "\n")
      })
      
      tryCatch({
        p3 <- easy_plot_residuals(model)
        resid_plot_path <- file.path(plots_dir, paste0("residuals_", timestamp, ".png"))
        ggplot2::ggsave(resid_plot_path, p3, width = 8, height = 6, dpi = 300)
        saved_files$plot_residuals <- resid_plot_path
        cat("  ✓ Residuals plot\n")
      }, error = function(e) {
        cat("  ⚠ Could not save residuals plot:", e$message, "\n")
      })
      
      tryCatch({
        p4 <- easy_plot_residual_dist(model)
        dist_plot_path <- file.path(plots_dir, paste0("residuals_dist_", timestamp, ".png"))
        ggplot2::ggsave(dist_plot_path, p4, width = 8, height = 6, dpi = 300)
        saved_files$plot_residual_dist <- dist_plot_path
        cat("  ✓ Residuals distribution plot\n")
      }, error = function(e) {
        cat("  ⚠ Could not save residuals distribution:", e$message, "\n")
      })
    }
  }
  
  # ============================================================================
  # SAVE SUMMARY REPORT
  # ============================================================================
  
  if (save_logs) {
    cat("\n📝 Saving summary report...\n")
    
    summary_path <- file.path(logs_dir, paste0("summary_report_", timestamp, ".txt"))
    
    summary_lines <- c(
      "╔════════════════════════════════════════════════════════════════════╗",
      "║              easyNNR MODEL - SUMMARY REPORT                        ║",
      "╚════════════════════════════════════════════════════════════════════╝",
      "",
      paste("Report generated:", as.character(Sys.time())),
      paste("Model saved in:", output_dir),
      "",
      "═══════════════════════════════════════════════════════════════════",
      "DATA SUMMARY",
      "═══════════════════════════════════════════════════════════════════",
      paste("Task:", toupper(model$parameters$task)),
      paste("Target variable:", model$target),
      paste("Input features:", model$parameters$input_dim),
      "",
      "═══════════════════════════════════════════════════════════════════",
      "MODEL ARCHITECTURE",
      "═══════════════════════════════════════════════════════════════════",
      paste("Layers:", paste(model$parameters$layers, collapse = " → ")),
      paste("Activations:", paste(model$parameters$activations, collapse = ", ")),
      paste("Batch Normalization:", model$parameters$batch_norm),
      paste("Dropout:", ifelse(is.null(model$parameters$dropout), "None", 
                               paste(model$parameters$dropout, collapse = ", "))),
      "",
      "═══════════════════════════════════════════════════════════════════",
      "TRAINING CONFIGURATION",
      "═══════════════════════════════════════════════════════════════════",
      paste("Optimizer:", model$parameters$optimizer),
      paste("Learning Rate:", model$parameters$learning_rate),
      paste("Epochs:", model$parameters$epochs),
      paste("Batch Size:", model$parameters$batch_size),
      paste("Early Stopping:", model$parameters$early_stopping),
      paste("Test Split:", model$parameters$test_split),
      "",
      "═══════════════════════════════════════════════════════════════════",
      "PERFORMANCE METRICS",
      "═══════════════════════════════════════════════════════════════════"
    )
    
    if (model$parameters$task == "classification") {
      summary_lines <- c(summary_lines,
                         paste("Accuracy:", round(model$evaluation$accuracy * 100, 2), "%"),
                         paste("Precision:", round(model$evaluation$precision * 100, 2), "%"),
                         paste("Recall:", round(model$evaluation$recall * 100, 2), "%"),
                         paste("F1-Score:", round(model$evaluation$f1_score, 4)),
                         paste("Loss:", round(model$evaluation$loss, 4))
      )
    } else {
      summary_lines <- c(summary_lines,
                         paste("R-squared:", round(model$evaluation$r_squared, 4)),
                         paste("RMSE:", round(model$evaluation$rmse, 2)),
                         paste("MAE:", round(model$evaluation$mae, 2)),
                         paste("Loss:", round(model$evaluation$loss, 4))
      )
    }
    
    summary_lines <- c(summary_lines,
                       "",
                       "═══════════════════════════════════════════════════════════════════",
                       "FILES SAVED",
                       "═══════════════════════════════════════════════════════════════════"
    )
    
    for (name in names(saved_files)) {
      summary_lines <- c(summary_lines, paste("•", name, ":", basename(saved_files[[name]])))
    }
    
    writeLines(summary_lines, summary_path)
    saved_files$summary <- summary_path
    cat("  ✓ Summary report:", summary_path, "\n")
  }
  
  # ============================================================================
  # FINAL MESSAGE
  # ============================================================================
  
  cat("\n✅ All files saved successfully!\n")
  cat("📁 Output directory:", output_dir, "\n")
  cat("📊 Total files saved:", length(saved_files), "\n\n")
  
  invisible(saved_files)
}


#' Load easyNNR Model
#'
#' @description
#' Load a previously saved easyNNR model from an RDS file.
#'
#' @param path Path to the saved model (.rds file)
#'
#' @return An easyNNR model object
#' @export
#'
#' @examples
#' \dontrun{
#' model <- load_easyNNR_model("~/my_models/models/latest_model.rds")
#' }
load_easyNNR_model <- function(path) {
  
  if (!file.exists(path)) {
    stop("Model file not found: ", path)
  }
  
  cat("📂 Loading easyNNR model from:", path, "\n")
  
  model <- readRDS(path)
  
  if (!inherits(model, "easyNNR")) {
    stop("Loaded object is not an easyNNR model")
  }
  
  cat("✓ Model loaded successfully!\n")
  cat("  Task:", model$parameters$task, "\n")
  cat("  Target:", model$target, "\n")
  
  if (model$parameters$task == "classification") {
    cat("  Accuracy:", round(model$evaluation$accuracy * 100, 2), "%\n")
  } else {
    cat("  R-squared:", round(model$evaluation$r_squared, 4), "\n")
  }
  
  cat("\n")
  
  return(model)
}


#' Save Training Session
#'
#' @description
#' Start a training session that logs all output to a file.
#'
#' @param log_dir Directory to save log files
#'
#' @return Connection object (for internal use)
#' @export
#'
#' @examples
#' \dontrun{
#' start_training_session("~/my_logs")
#' model <- easy_nn(iris, target = "Species")
#' end_training_session()
#' }
start_training_session <- function(log_dir) {
  
  dir.create(log_dir, showWarnings = FALSE, recursive = TRUE)
  
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  log_file <- file.path(log_dir, paste0("training_session_", timestamp, ".log"))
  
  log_con <- file(log_file, open = "wt")
  sink(log_con, type = "output", split = TRUE)
  sink(log_con, type = "message")
  
  cat("═══════════════════════════════════════════════════════════════════\n")
  cat("easyNNR Training Session Started\n")
  cat("═══════════════════════════════════════════════════════════════════\n")
  cat("Time:", as.character(Sys.time()), "\n")
  cat("Log file:", log_file, "\n")
  cat("═══════════════════════════════════════════════════════════════════\n\n")
  
  # Store in package environment for later retrieval
  .easyNNR_env$log_connection <- log_con
  .easyNNR_env$log_file <- log_file
  
  invisible(log_con)
}


#' End Training Session
#'
#' @description
#' End a training session and close the log file.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' start_training_session("~/my_logs")
#' model <- easy_nn(iris, target = "Species")
#' end_training_session()
#' }
end_training_session <- function() {
  
  if (exists("log_connection", envir = .easyNNR_env)) {
    cat("\n═══════════════════════════════════════════════════════════════════\n")
    cat("easyNNR Training Session Ended\n")
    cat("Time:", as.character(Sys.time()), "\n")
    cat("═══════════════════════════════════════════════════════════════════\n")
    
    sink(type = "output")
    sink(type = "message")
    close(.easyNNR_env$log_connection)
    
    cat("✓ Session log saved:", .easyNNR_env$log_file, "\n\n")
    
    rm(log_connection, log_file, envir = .easyNNR_env)
  } else {
    message("No active training session to end")
  }
}


# Create package environment for storing session data
.easyNNR_env <- new.env(parent = emptyenv())