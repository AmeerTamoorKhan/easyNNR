#' Predict with an easyNNR Model
#'
#' @description
#' Make predictions on new data using a trained easyNNR model.
#' Automatically applies the same preprocessing pipeline used during training.
#'
#' @param object An 'easyNNR' object from easy_nn()
#' @param new_data A data frame with the same schema as training data
#'   (can include excluded columns - they will be ignored)
#' @param type Character string: "class" for class labels (classification),
#'   "prob" for probabilities (classification), or "response" for numeric values (regression).
#'   Default: "class" for classification, "response" for regression.
#' @param verbose Logical, whether to print progress messages (default: TRUE)
#'
#' @return
#' For regression: Numeric vector of predictions (on original scale if target was transformed)
#' For classification:
#'   - type = "class": Factor with predicted class labels
#'   - type = "prob": Matrix of class probabilities (one column per class)
#'
#' @examples
#' \dontrun{
#' # Train model
#' model <- easy_nn(iris, target = "Species")
#' 
#' # Predict on new data
#' new_flowers <- iris[1:10, ]
#' predictions <- easy_predict(model, new_flowers)
#' 
#' # Get probabilities instead
#' probs <- easy_predict(model, new_flowers, type = "prob")
#' 
#' # Regression example with target transformation
#' model_reg <- easy_nn(
#'   mtcars, 
#'   target = "mpg", 
#'   task = "regression",
#'   preprocess = list(target_transform = "log")
#' )
#' new_cars <- mtcars[1:5, ]
#' mpg_predictions <- easy_predict(model_reg, new_cars)  # Returns original scale
#' }
#'
#' @export
easy_predict <- function(object, new_data, type = NULL, verbose = TRUE) {
  
  # ============================================================================
  # Input Validation
  # ============================================================================
  
  if (!inherits(object, "easyNNR")) {
    stop("'object' must be an easyNNR model from easy_nn()", call. = FALSE)
  }
  
  if (!is.data.frame(new_data)) {
    stop("'new_data' must be a data.frame or tibble", call. = FALSE)
  }
  
  if (nrow(new_data) == 0) {
    stop("'new_data' has no rows", call. = FALSE)
  }
  
  task <- object$parameters$task
  
  # Set default type based on task
  if (is.null(type)) {
    type <- if (task == "classification") "class" else "response"
  }
  
  # Validate type parameter
  valid_types <- c("class", "prob", "response")
  if (!type %in% valid_types) {
    stop("'type' must be one of: ", paste(valid_types, collapse = ", "), call. = FALSE)
  }
  
  if (task == "regression" && type %in% c("class", "prob")) {
    warning("For regression tasks, type is set to 'response'")
    type <- "response"
  }
  
  if (verbose) {
    cat("\n🔮 Generating predictions...\n")
    cat("   Samples:", nrow(new_data), "\n")
    cat("   Task:", task, "\n")
  }
  
  # ============================================================================
  # Apply Preprocessing
  # ============================================================================
  
  if (verbose) cat("   Applying preprocessing pipeline...\n")
  
  # Check if time series features need to be extracted
  preprocess_opts <- object$preprocessing$options
  
  # Extract date features if model was trained with them
  if (!is.null(preprocess_opts) && isTRUE(preprocess_opts$date_features)) {
    date_cols <- .identify_date_columns(new_data)
    if (length(date_cols) > 0) {
      new_data <- .extract_date_features(new_data, date_cols)
    }
  }
  
  # Apply stored recipe directly
  baked <- tryCatch(
    recipes::bake(object$recipe, new_data = new_data),
    error = function(e) {
      stop(
        "❌ Failed to apply preprocessing to new_data.\n",
        "   Make sure new_data has the same columns or factor levels as training data.\n",
        "   Error: ", conditionMessage(e), "\n",
        call. = FALSE
      )
    }
  )
  
  # Remove target if present
  if (object$target %in% names(baked)) {
    baked <- baked[, setdiff(names(baked), object$target), drop = FALSE]
  }
  
  # Apply feature selection if it was used during training
  selected_features <- object$preprocessing$selected_features
  if (!is.null(selected_features)) {
    # Keep only selected features (that exist in baked)
    available <- intersect(selected_features, names(baked))
    if (length(available) < length(selected_features)) {
      missing <- setdiff(selected_features, names(baked))
      warning("Some selected features not found in new_data: ", 
              paste(head(missing, 5), collapse = ", "))
    }
    baked <- baked[, available, drop = FALSE]
  }
  
  X <- as.matrix(baked)
  
  # Handle any remaining NAs
  if (any(is.na(X))) {
    if (verbose) cat("   ⚠️  Handling", sum(is.na(X)), "missing values in features...\n")
    for (j in seq_len(ncol(X))) {
      if (any(is.na(X[, j]))) {
        X[is.na(X[, j]), j] <- mean(X[, j], na.rm = TRUE)
      }
    }
  }
  
  # ============================================================================
  # Make Predictions
  # ============================================================================
  
  if (verbose) cat("   Computing predictions...\n")
  
  model <- object$model
  
  # Get raw predictions from model
  predictions_raw <- model(X, training = FALSE) |> as.matrix()
  
  # ============================================================================
  # Format Predictions Based on Task and Type
  # ============================================================================
  
  if (task == "regression") {
    # Regression: return numeric vector
    result <- as.numeric(predictions_raw[, 1])
    
    # Inverse transform if target was transformed during training
    target_transformer <- object$preprocessing$target_transformer
    if (!is.null(target_transformer)) {
      result <- .inverse_transform_target(result, target_transformer)
      if (verbose) {
        cat("   ✓ Applied inverse target transformation\n")
      }
    }
    
    if (verbose) {
      cat("   ✓ Predictions complete\n")
      cat("   Range: [", round(min(result), 2), ", ", 
          round(max(result), 2), "]\n", sep = "")
    }
    
  } else {
    # Classification
    class_levels <- object$parameters$class_levels
    num_classes <- object$parameters$num_classes
    
    if (num_classes == 2) {
      # Binary classification
      probs_positive <- as.numeric(predictions_raw[, 1])
      
      if (type == "prob") {
        # Return probability matrix
        result <- cbind(
          1 - probs_positive,  # Probability of first class
          probs_positive       # Probability of second class
        )
        colnames(result) <- class_levels
        
      } else {
        # Return class labels
        predicted_labels <- ifelse(
          probs_positive >= 0.5,
          class_levels[2],
          class_levels[1]
        )
        result <- factor(predicted_labels, levels = class_levels)
      }
      
    } else {
      # Multi-class classification
      
      if (type == "prob") {
        # Return probability matrix
        result <- predictions_raw
        colnames(result) <- class_levels
        
      } else {
        # Return class labels
        predicted_indices <- max.col(predictions_raw)
        result <- factor(class_levels[predicted_indices], levels = class_levels)
      }
    }
    
    if (verbose) {
      cat("   ✓ Predictions complete\n")
      if (type == "class") {
        cat("   Predicted classes:\n")
        print(table(result))
      } else {
        cat("   Returning probability matrix:", nrow(result), "×", ncol(result), "\n")
      }
    }
  }
  
  if (verbose) cat("\n")
  
  return(result)
}


#' Predict Method for easyNNR Objects
#'
#' @description
#' S3 method for predict() generic. Calls easy_predict() internally.
#'
#' @param object An easyNNR model object
#' @param newdata New data to make predictions on
#' @param type Type of prediction ("class", "prob", or "response")
#' @param ... Additional arguments (currently unused)
#'
#' @return Predictions (format depends on type)
#'
#' @export
predict.easyNNR <- function(object, newdata, type = NULL, ...) {
  easy_predict(object, new_data = newdata, type = type, verbose = FALSE)
}


#' Batch Prediction with Progress
#'
#' @description
#' Make predictions on large datasets in batches to manage memory.
#'
#' @param object An easyNNR model object
#' @param new_data Data frame to predict on
#' @param batch_size Number of samples per batch (default: 1000)
#' @param type Prediction type
#' @param verbose Whether to show progress
#'
#' @return Predictions
#'
#' @export
easy_predict_batch <- function(object, new_data, batch_size = 1000, 
                                type = NULL, verbose = TRUE) {
  
  n <- nrow(new_data)
  n_batches <- ceiling(n / batch_size)
  
  results <- vector("list", n_batches)
  
  if (verbose) {
    cat("Predicting", n, "samples in", n_batches, "batches...\n")
  }
  
  for (i in seq_len(n_batches)) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, n)
    
    batch_data <- new_data[start_idx:end_idx, , drop = FALSE]
    results[[i]] <- easy_predict(object, batch_data, type = type, verbose = FALSE)
    
    if (verbose) {
      cat("\r  Batch", i, "/", n_batches, "complete")
    }
  }
  
  if (verbose) cat("\n")
  
  # Combine results
  if (is.matrix(results[[1]])) {
    do.call(rbind, results)
  } else if (is.factor(results[[1]])) {
    factor(unlist(lapply(results, as.character)), 
           levels = levels(results[[1]]))
  } else {
    unlist(results)
  }
}


#' Get Prediction Confidence
#'
#' @description
#' For classification models, returns the confidence (probability) of the
#' predicted class for each sample.
#'
#' @param object An easyNNR classification model
#' @param new_data Data frame to predict on
#'
#' @return Data frame with predictions and confidence scores
#'
#' @export
easy_predict_confidence <- function(object, new_data) {
  
  if (object$parameters$task != "classification") {
    stop("Confidence scores are only available for classification models")
  }
  
  # Get probabilities
  probs <- easy_predict(object, new_data, type = "prob", verbose = FALSE)
  
  # Get predicted classes
  classes <- easy_predict(object, new_data, type = "class", verbose = FALSE)
  
  # Get confidence (max probability)
  confidence <- apply(probs, 1, max)
  
  data.frame(
    predicted = classes,
    confidence = confidence,
    row.names = NULL
  )
}