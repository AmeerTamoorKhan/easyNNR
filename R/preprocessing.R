#' Preprocessing Utilities for easyNNR
#'
#' @description
#' Internal functions for advanced preprocessing including outlier handling,
#' target transformation, class imbalance, feature selection, and more.
#'
#' @name preprocessing
#' @keywords internal
NULL


# ==============================================================================
# PREPROCESSING OPTIONS PARSER
# ==============================================================================

#' Parse and Validate Preprocessing Options
#' @param opts User-provided preprocessing options list
#' @return Complete list with defaults filled in
#' @keywords internal
.parse_preprocess_options <- function(opts) {
  defaults <- list(
    # Outlier handling
    outlier_method = "none",
    outlier_threshold = NULL,
    
    # Target transformation
    target_transform = "none",
    
    # Class imbalance
    imbalance_method = "none",
    imbalance_ratio = 1.0,
    
    # Feature selection
    feature_selection = "none",
    n_features = NULL,
    correlation_threshold = 0.9,
    
    # Encoding
    encoding = "onehot",
    max_categories = 50,
    
    # Imputation
    impute_numeric = "median",
    impute_categorical = "mode",
    add_indicators = FALSE,
    
    # Feature engineering
    interactions = FALSE,
    polynomial_degree = 1,
    pca_components = NULL,
    
    # Time series
    date_features = FALSE,
    lag_features = NULL,
    rolling_features = NULL
  )
  
  # Merge user options with defaults
  for (name in names(opts)) {
    if (name %in% names(defaults)) {
      defaults[[name]] <- opts[[name]]
    } else {
      warning("Unknown preprocessing option '", name, "' - ignoring.", call. = FALSE)
    }
  }
  
  # Set default outlier thresholds based on method
  if (is.null(defaults$outlier_threshold)) {
    defaults$outlier_threshold <- switch(
      defaults$outlier_method,
      "iqr" = 1.5,
      "zscore" = 3,
      "isolation_forest" = 0.1,
      "winsorize" = 0.05,
      "cap" = 1.5,
      1.5
    )
  }
  
  # Validate options
  valid_outlier <- c("none", "iqr", "zscore", "isolation_forest", "winsorize", "cap", "remove")
  if (!defaults$outlier_method %in% valid_outlier) {
    stop("Invalid outlier_method. Must be one of: ", paste(valid_outlier, collapse = ", "))
  }
  
  valid_transform <- c("none", "log", "log1p", "sqrt", "boxcox", "yeojohnson", "quantile")
  if (!defaults$target_transform %in% valid_transform) {
    stop("Invalid target_transform. Must be one of: ", paste(valid_transform, collapse = ", "))
  }
  
  valid_imbalance <- c("none", "oversample", "undersample", "smote", "adasyn", "class_weights")
  if (!defaults$imbalance_method %in% valid_imbalance) {
    stop("Invalid imbalance_method. Must be one of: ", paste(valid_imbalance, collapse = ", "))
  }
  
  valid_selection <- c("none", "variance", "correlation", "mutual_info", "rfe", "lasso")
  if (!defaults$feature_selection %in% valid_selection) {
    stop("Invalid feature_selection. Must be one of: ", paste(valid_selection, collapse = ", "))
  }
  
  valid_encoding <- c("onehot", "target", "frequency", "binary", "hash")
  if (!defaults$encoding %in% valid_encoding) {
    stop("Invalid encoding. Must be one of: ", paste(valid_encoding, collapse = ", "))
  }
  
  defaults
}


# ==============================================================================
# OUTLIER HANDLING
# ==============================================================================

#' Handle Outliers in Numeric Features
#' @param data Data frame
#' @param features Character vector of feature names
#' @param method Outlier handling method
#' @param threshold Threshold for detection
#' @return List with modified data and info
#' @keywords internal
.handle_outliers <- function(data, features, method, threshold) {
  
  numeric_features <- features[sapply(data[, features, drop = FALSE], is.numeric)]
  
  if (length(numeric_features) == 0) {
    return(list(data = data, info = list(n_affected = 0)))
  }
  
  n_affected <- 0
  outlier_info <- list()
  
  for (feat in numeric_features) {
    x <- data[[feat]]
    na_mask <- is.na(x)
    x_valid <- x[!na_mask]
    
    if (length(x_valid) == 0) next
    
    # Detect outliers based on method
    outlier_mask <- switch(
      method,
      
      "iqr" = {
        q1 <- quantile(x_valid, 0.25)
        q3 <- quantile(x_valid, 0.75)
        iqr <- q3 - q1
        lower <- q1 - threshold * iqr
        upper <- q3 + threshold * iqr
        x < lower | x > upper
      },
      
      "zscore" = {
        z <- abs((x - mean(x_valid)) / sd(x_valid))
        z > threshold
      },
      
      "isolation_forest" = {
        # Simple percentile-based approximation
        lower_pct <- threshold / 2
        upper_pct <- 1 - threshold / 2
        x < quantile(x_valid, lower_pct) | x > quantile(x_valid, upper_pct)
      },
      
      rep(FALSE, length(x))
    )
    
    outlier_mask[na_mask] <- FALSE
    n_outliers <- sum(outlier_mask, na.rm = TRUE)
    
    if (n_outliers == 0) next
    
    n_affected <- n_affected + n_outliers
    outlier_info[[feat]] <- list(n_outliers = n_outliers, method = method)
    
    # Handle outliers based on method
    if (method %in% c("iqr", "zscore", "isolation_forest", "cap")) {
      # Cap at boundaries
      q1 <- quantile(x_valid, 0.25)
      q3 <- quantile(x_valid, 0.75)
      iqr <- q3 - q1
      lower <- q1 - threshold * iqr
      upper <- q3 + threshold * iqr
      
      data[[feat]][x < lower & !na_mask] <- lower
      data[[feat]][x > upper & !na_mask] <- upper
      
    } else if (method == "winsorize") {
      # Winsorization: replace with percentile values
      lower_val <- quantile(x_valid, threshold)
      upper_val <- quantile(x_valid, 1 - threshold)
      
      data[[feat]][x < lower_val & !na_mask] <- lower_val
      data[[feat]][x > upper_val & !na_mask] <- upper_val
      
    } else if (method == "remove") {
      # Mark for removal (handled after loop)
      if (!exists("rows_to_remove")) rows_to_remove <- rep(FALSE, nrow(data))
      rows_to_remove <- rows_to_remove | outlier_mask
    }
  }
  
  # Remove rows if method is "remove"
  if (method == "remove" && exists("rows_to_remove")) {
    data <- data[!rows_to_remove, , drop = FALSE]
  }
  
  list(
    data = data,
    info = list(
      n_affected = n_affected,
      by_feature = outlier_info,
      method = method,
      threshold = threshold
    )
  )
}


# ==============================================================================
# TARGET TRANSFORMATION
# ==============================================================================

#' Transform Target Variable (Regression)
#' @param y Numeric target vector
#' @param method Transformation method
#' @return List with transformed values and transformer info
#' @keywords internal
.transform_target <- function(y, method) {
  
  transformer <- list(method = method)
  
  transformed <- switch(
    method,
    
    "log" = {
      if (any(y <= 0, na.rm = TRUE)) {
        stop("Target contains non-positive values. Use 'log1p' instead of 'log'.")
      }
      transformer$offset <- 0
      log(y)
    },
    
    "log1p" = {
      if (any(y < 0, na.rm = TRUE)) {
        # Add offset to make all values positive
        transformer$offset <- abs(min(y, na.rm = TRUE)) + 1
        log1p(y + transformer$offset)
      } else {
        transformer$offset <- 0
        log1p(y)
      }
    },
    
    "sqrt" = {
      if (any(y < 0, na.rm = TRUE)) {
        transformer$offset <- abs(min(y, na.rm = TRUE)) + 1
        sqrt(y + transformer$offset)
      } else {
        transformer$offset <- 0
        sqrt(y)
      }
    },
    
    "boxcox" = {
      if (any(y <= 0, na.rm = TRUE)) {
        transformer$offset <- abs(min(y, na.rm = TRUE)) + 1
        y_shifted <- y + transformer$offset
      } else {
        transformer$offset <- 0
        y_shifted <- y
      }
      
      # Find optimal lambda using Box-Cox
      bc <- .boxcox_transform(y_shifted)
      transformer$lambda <- bc$lambda
      bc$transformed
    },
    
    "yeojohnson" = {
      yj <- .yeojohnson_transform(y)
      transformer$lambda <- yj$lambda
      yj$transformed
    },
    
    "quantile" = {
      # Quantile transformation to uniform/normal distribution
      transformer$quantiles <- ecdf(y)
      transformer$values <- sort(y)
      qnorm(rank(y) / (length(y) + 1))
    },
    
    y  # Default: no transformation
  )
  
  list(transformed = transformed, transformer = transformer)
}


#' Inverse Transform Target Variable
#' @param y_transformed Transformed target values
#' @param transformer Transformer object from .transform_target
#' @return Original scale values
#' @keywords internal
.inverse_transform_target <- function(y_transformed, transformer) {
  
  method <- transformer$method
  
  switch(
    method,
    
    "log" = exp(y_transformed),
    
    "log1p" = expm1(y_transformed) - transformer$offset,
    
    "sqrt" = y_transformed^2 - transformer$offset,
    
    "boxcox" = {
      lambda <- transformer$lambda
      if (abs(lambda) < 1e-6) {
        exp(y_transformed)
      } else {
        (y_transformed * lambda + 1)^(1/lambda)
      }
      - transformer$offset
    },
    
    "yeojohnson" = {
      .yeojohnson_inverse(y_transformed, transformer$lambda)
    },
    
    "quantile" = {
      # Inverse quantile transformation
      probs <- pnorm(y_transformed)
      probs <- pmax(pmin(probs, 0.999), 0.001)
      quantile(transformer$values, probs)
    },
    
    y_transformed  # Default: no transformation
  )
}


#' Box-Cox Transformation
#' @keywords internal
.boxcox_transform <- function(y) {
  # Grid search for optimal lambda
  lambdas <- seq(-2, 2, by = 0.1)
  log_likelihoods <- sapply(lambdas, function(lambda) {
    if (abs(lambda) < 1e-6) {
      transformed <- log(y)
    } else {
      transformed <- (y^lambda - 1) / lambda
    }
    n <- length(y)
    -n/2 * log(var(transformed)) + (lambda - 1) * sum(log(y))
  })
  
  best_lambda <- lambdas[which.max(log_likelihoods)]
  
  if (abs(best_lambda) < 1e-6) {
    transformed <- log(y)
  } else {
    transformed <- (y^best_lambda - 1) / best_lambda
  }
  
  list(transformed = transformed, lambda = best_lambda)
}


#' Yeo-Johnson Transformation
#' @keywords internal
.yeojohnson_transform <- function(y) {
  # Grid search for optimal lambda
  lambdas <- seq(-2, 2, by = 0.1)
  
  yj_transform <- function(y, lambda) {
    transformed <- y
    pos_mask <- y >= 0
    
    if (abs(lambda) < 1e-6) {
      transformed[pos_mask] <- log1p(y[pos_mask])
    } else {
      transformed[pos_mask] <- ((y[pos_mask] + 1)^lambda - 1) / lambda
    }
    
    if (abs(lambda - 2) < 1e-6) {
      transformed[!pos_mask] <- -log1p(-y[!pos_mask])
    } else {
      transformed[!pos_mask] <- -((-y[!pos_mask] + 1)^(2 - lambda) - 1) / (2 - lambda)
    }
    
    transformed
  }
  
  variances <- sapply(lambdas, function(l) var(yj_transform(y, l)))
  best_lambda <- lambdas[which.min(variances)]
  
  list(transformed = yj_transform(y, best_lambda), lambda = best_lambda)
}


#' Yeo-Johnson Inverse Transformation
#' @keywords internal
.yeojohnson_inverse <- function(y, lambda) {
  result <- y
  
  # This is approximate - full inverse requires knowing original sign
  if (abs(lambda) < 1e-6) {
    result <- expm1(y)
  } else if (abs(lambda - 2) < 1e-6) {
    result <- 1 - exp(-y)
  } else {
    # Positive branch approximation
    result <- (y * lambda + 1)^(1/lambda) - 1
  }
  
  result
}


# ==============================================================================
# CLASS IMBALANCE HANDLING
# ==============================================================================

#' Calculate Class Weights
#' @param y Factor or character target
#' @param class_levels Character vector of class levels
#' @return Named list of class weights
#' @keywords internal
.calculate_class_weights <- function(y, class_levels) {
  class_counts <- table(y)
  n_samples <- length(y)
  n_classes <- length(class_levels)
  
  # Balanced class weights: n_samples / (n_classes * n_class_samples)
  weights <- n_samples / (n_classes * class_counts)
  
  # Convert to named list with integer indices (0-based for Keras)
  weight_list <- as.list(as.numeric(weights))
  names(weight_list) <- as.character(0:(n_classes - 1))
  
  weight_list
}


#' Handle Class Imbalance via Resampling
#' @param X Feature matrix
#' @param y Integer target vector (0-indexed class labels)
#' @param method Resampling method
#' @param ratio Target ratio of minority to majority
#' @param k Number of neighbors for SMOTE/ADASYN
#' @return List with resampled X and y
#' @keywords internal
.handle_imbalance <- function(X, y, method, ratio = 1.0, k = 5) {
  
  class_counts <- table(y)
  majority_class <- as.integer(names(which.max(class_counts)))
  minority_classes <- as.integer(names(class_counts)[names(class_counts) != as.character(majority_class)])
  
  n_majority <- max(class_counts)
  
  if (method == "oversample") {
    # Random oversampling of minority classes
    X_new <- X
    y_new <- y
    
    for (mc in minority_classes) {
      mc_idx <- which(y == mc)
      n_to_add <- round(n_majority * ratio) - length(mc_idx)
      
      if (n_to_add > 0) {
        # Sample with replacement
        sampled_idx <- sample(mc_idx, n_to_add, replace = TRUE)
        X_new <- rbind(X_new, X[sampled_idx, , drop = FALSE])
        y_new <- c(y_new, y[sampled_idx])
      }
    }
    
    return(list(X = X_new, y = y_new))
    
  } else if (method == "undersample") {
    # Random undersampling of majority class
    target_n <- round(min(class_counts) / ratio)
    
    idx_keep <- c()
    for (cl in unique(y)) {
      cl_idx <- which(y == cl)
      n_keep <- min(length(cl_idx), target_n)
      idx_keep <- c(idx_keep, sample(cl_idx, n_keep))
    }
    
    return(list(X = X[idx_keep, , drop = FALSE], y = y[idx_keep]))
    
  } else if (method == "smote") {
    # SMOTE: Synthetic Minority Over-sampling Technique
    return(.smote_resample(X, y, k = k, ratio = ratio))
    
  } else if (method == "adasyn") {
    # ADASYN: Adaptive Synthetic Sampling
    return(.adasyn_resample(X, y, k = k, ratio = ratio))
  }
  
  list(X = X, y = y)
}


#' SMOTE Resampling
#' @keywords internal
.smote_resample <- function(X, y, k = 5, ratio = 1.0) {
  
  class_counts <- table(y)
  majority_class <- as.integer(names(which.max(class_counts)))
  n_majority <- max(class_counts)
  
  X_new <- X
  y_new <- y
  
  for (mc in unique(y)) {
    if (mc == majority_class) next
    
    mc_idx <- which(y == mc)
    n_current <- length(mc_idx)
    n_target <- round(n_majority * ratio)
    n_to_generate <- n_target - n_current
    
    if (n_to_generate <= 0) next
    
    X_minority <- X[mc_idx, , drop = FALSE]
    
    # Generate synthetic samples
    synthetic_X <- matrix(0, nrow = n_to_generate, ncol = ncol(X))
    
    for (i in seq_len(n_to_generate)) {
      # Pick a random minority sample
      idx <- sample(nrow(X_minority), 1)
      sample_point <- X_minority[idx, ]
      
      # Find k nearest neighbors among minority class
      distances <- apply(X_minority, 1, function(row) sum((row - sample_point)^2))
      nn_idx <- order(distances)[2:min(k + 1, nrow(X_minority))]
      
      if (length(nn_idx) == 0) nn_idx <- 1
      
      # Pick a random neighbor
      neighbor_idx <- sample(nn_idx, 1)
      neighbor <- X_minority[neighbor_idx, ]
      
      # Generate synthetic sample along the line between sample and neighbor
      alpha <- runif(1)
      synthetic_X[i, ] <- sample_point + alpha * (neighbor - sample_point)
    }
    
    X_new <- rbind(X_new, synthetic_X)
    y_new <- c(y_new, rep(mc, n_to_generate))
  }
  
  list(X = X_new, y = y_new)
}


#' ADASYN Resampling
#' @keywords internal
.adasyn_resample <- function(X, y, k = 5, ratio = 1.0) {
  
  class_counts <- table(y)
  majority_class <- as.integer(names(which.max(class_counts)))
  n_majority <- max(class_counts)
  
  X_new <- X
  y_new <- y
  
  for (mc in unique(y)) {
    if (mc == majority_class) next
    
    mc_idx <- which(y == mc)
    n_current <- length(mc_idx)
    n_target <- round(n_majority * ratio)
    G <- n_target - n_current  # Total synthetic samples needed
    
    if (G <= 0) next
    
    X_minority <- X[mc_idx, , drop = FALSE]
    
    # Calculate ratio of majority neighbors for each minority sample
    ratios <- numeric(nrow(X_minority))
    
    for (i in seq_len(nrow(X_minority))) {
      sample_point <- X_minority[i, ]
      
      # Find k nearest neighbors in entire dataset
      distances <- apply(X, 1, function(row) sum((row - sample_point)^2))
      nn_idx <- order(distances)[2:min(k + 1, nrow(X))]
      
      # Count majority class neighbors
      n_majority_neighbors <- sum(y[nn_idx] == majority_class)
      ratios[i] <- n_majority_neighbors / k
    }
    
    # Normalize ratios
    if (sum(ratios) > 0) {
      ratios <- ratios / sum(ratios)
    } else {
      ratios <- rep(1/length(ratios), length(ratios))
    }
    
    # Number of synthetic samples per minority sample
    g_per_sample <- round(ratios * G)
    
    # Generate synthetic samples
    for (i in seq_len(nrow(X_minority))) {
      n_gen <- g_per_sample[i]
      if (n_gen == 0) next
      
      sample_point <- X_minority[i, ]
      
      # Find k nearest neighbors among minority class
      distances <- apply(X_minority, 1, function(row) sum((row - sample_point)^2))
      nn_idx <- order(distances)[2:min(k + 1, nrow(X_minority))]
      
      if (length(nn_idx) == 0) nn_idx <- 1
      
      for (j in seq_len(n_gen)) {
        neighbor_idx <- sample(nn_idx, 1)
        neighbor <- X_minority[neighbor_idx, ]
        
        alpha <- runif(1)
        synthetic <- sample_point + alpha * (neighbor - sample_point)
        
        X_new <- rbind(X_new, synthetic)
        y_new <- c(y_new, mc)
      }
    }
  }
  
  list(X = X_new, y = y_new)
}


# ==============================================================================
# FEATURE SELECTION
# ==============================================================================

#' Select Features Using Various Methods
#' @param X Feature matrix
#' @param y Target vector
#' @param method Selection method
#' @param n_features Number of features to select
#' @param task "classification" or "regression"
#' @return List with selected feature indices and selector info
#' @keywords internal
.select_features <- function(X, y, method, n_features = NULL, task = "classification") {
  
  p <- ncol(X)
  
  # Default to half of features if not specified
  if (is.null(n_features)) {
    n_features <- max(1, floor(p / 2))
  }
  
  n_features <- min(n_features, p)
  
  selector <- list(method = method, n_features = n_features)
  
  if (method == "mutual_info") {
    # Mutual information-based selection
    scores <- .mutual_info_scores(X, y, task)
    selected_idx <- order(scores, decreasing = TRUE)[seq_len(n_features)]
    selector$scores <- scores
    
  } else if (method == "lasso") {
    # L1-based feature selection
    selected_idx <- .lasso_selection(X, y, n_features, task)
    
  } else if (method == "rfe") {
    # Recursive feature elimination (simplified)
    selected_idx <- .rfe_selection(X, y, n_features, task)
    
  } else {
    # Default: use all features
    selected_idx <- seq_len(p)
  }
  
  # Get feature names if available
  selected_names <- if (!is.null(colnames(X))) {
    colnames(X)[selected_idx]
  } else {
    paste0("V", selected_idx)
  }
  
  list(
    selected = selected_names,
    indices = selected_idx,
    selector = selector
  )
}


#' Calculate Mutual Information Scores
#' @keywords internal
.mutual_info_scores <- function(X, y, task) {
  
  n <- nrow(X)
  p <- ncol(X)
  scores <- numeric(p)
  
  for (j in seq_len(p)) {
    x_j <- X[, j]
    
    if (task == "classification") {
      # Discretize continuous features for MI calculation
      x_discrete <- cut(x_j, breaks = min(10, length(unique(x_j))), labels = FALSE)
      x_discrete[is.na(x_discrete)] <- 0
      
      # Calculate mutual information
      joint_freq <- table(x_discrete, y) / n
      x_freq <- table(x_discrete) / n
      y_freq <- table(y) / n
      
      mi <- 0
      for (i in seq_len(nrow(joint_freq))) {
        for (k in seq_len(ncol(joint_freq))) {
          if (joint_freq[i, k] > 0 && x_freq[i] > 0 && y_freq[k] > 0) {
            mi <- mi + joint_freq[i, k] * log(joint_freq[i, k] / (x_freq[i] * y_freq[k]))
          }
        }
      }
      scores[j] <- mi
      
    } else {
      # For regression: use correlation as proxy
      scores[j] <- abs(cor(x_j, y, use = "complete.obs"))
    }
  }
  
  scores
}


#' L1/LASSO-based Feature Selection
#' @keywords internal
.lasso_selection <- function(X, y, n_features, task) {
  
  # Simple implementation using correlation-weighted selection
  # (Full LASSO would require glmnet dependency)
  
  p <- ncol(X)
  
  if (task == "regression") {
    scores <- abs(apply(X, 2, function(x) cor(x, y, use = "complete.obs")))
  } else {
    # For classification: use ANOVA F-statistic
    scores <- apply(X, 2, function(x) {
      groups <- split(x, y)
      group_means <- sapply(groups, mean, na.rm = TRUE)
      overall_mean <- mean(x, na.rm = TRUE)
      
      between_var <- sum(sapply(seq_along(groups), function(i) {
        length(groups[[i]]) * (group_means[i] - overall_mean)^2
      }))
      
      within_var <- sum(sapply(groups, function(g) sum((g - mean(g))^2, na.rm = TRUE)))
      
      if (within_var > 0) between_var / within_var else 0
    })
  }
  
  scores[is.na(scores)] <- 0
  order(scores, decreasing = TRUE)[seq_len(n_features)]
}


#' Recursive Feature Elimination (Simplified)
#' @keywords internal
.rfe_selection <- function(X, y, n_features, task) {
  
  p <- ncol(X)
  remaining <- seq_len(p)
  
  while (length(remaining) > n_features) {
    X_subset <- X[, remaining, drop = FALSE]
    
    # Calculate importance scores
    if (task == "regression") {
      scores <- abs(apply(X_subset, 2, function(x) cor(x, y, use = "complete.obs")))
    } else {
      scores <- apply(X_subset, 2, function(x) {
        groups <- split(x, y)
        var_between <- var(sapply(groups, mean, na.rm = TRUE))
        var_within <- mean(sapply(groups, var, na.rm = TRUE))
        if (var_within > 0) var_between / var_within else 0
      })
    }
    
    scores[is.na(scores)] <- 0
    
    # Remove worst feature
    worst_idx <- which.min(scores)
    remaining <- remaining[-worst_idx]
  }
  
  remaining
}


# ==============================================================================
# TIME SERIES FEATURES
# ==============================================================================

#' Identify Date Columns
#' @keywords internal
.identify_date_columns <- function(data) {
  date_cols <- c()
  
  for (col in names(data)) {
    x <- data[[col]]
    
    if (inherits(x, c("Date", "POSIXt", "POSIXct", "POSIXlt"))) {
      date_cols <- c(date_cols, col)
    } else if (is.character(x) || is.factor(x)) {
      # Try to parse as date
      x_char <- as.character(x)[!is.na(x)][1:min(10, sum(!is.na(x)))]
      parsed <- suppressWarnings(as.Date(x_char))
      if (sum(!is.na(parsed)) > length(x_char) * 0.5) {
        date_cols <- c(date_cols, col)
      }
    }
  }
  
  date_cols
}


#' Extract Features from Date Columns
#' @keywords internal
.extract_date_features <- function(data, date_cols) {
  
  for (col in date_cols) {
    x <- data[[col]]
    
    # Convert to Date if needed
    if (!inherits(x, c("Date", "POSIXt"))) {
      x <- as.Date(as.character(x))
    }
    
    # Extract date components
    data[[paste0(col, "_year")]] <- as.integer(format(x, "%Y"))
    data[[paste0(col, "_month")]] <- as.integer(format(x, "%m"))
    data[[paste0(col, "_day")]] <- as.integer(format(x, "%d"))
    data[[paste0(col, "_dayofweek")]] <- as.integer(format(x, "%u"))  # 1=Monday
    data[[paste0(col, "_dayofyear")]] <- as.integer(format(x, "%j"))
    data[[paste0(col, "_quarter")]] <- as.integer(ceiling(as.integer(format(x, "%m")) / 3))
    data[[paste0(col, "_weekofyear")]] <- as.integer(format(x, "%V"))
    
    # Cyclical encoding for periodic features
    month_num <- as.integer(format(x, "%m"))
    data[[paste0(col, "_month_sin")]] <- sin(2 * pi * month_num / 12)
    data[[paste0(col, "_month_cos")]] <- cos(2 * pi * month_num / 12)
    
    dow_num <- as.integer(format(x, "%u"))
    data[[paste0(col, "_dow_sin")]] <- sin(2 * pi * dow_num / 7)
    data[[paste0(col, "_dow_cos")]] <- cos(2 * pi * dow_num / 7)
    
    # Is weekend
    data[[paste0(col, "_is_weekend")]] <- as.integer(dow_num %in% c(6, 7))
  }
  
  data
}


#' Create Lag Features
#' @keywords internal
.create_lag_features <- function(data, features, lags) {
  
  for (feat in features) {
    for (lag in lags) {
      new_name <- paste0(feat, "_lag", lag)
      data[[new_name]] <- dplyr::lag(data[[feat]], n = lag)
    }
  }
  
  data
}


#' Create Rolling Features
#' @keywords internal
.create_rolling_features <- function(data, features, rolling_config) {
  
  windows <- rolling_config$windows
  if (is.null(windows)) windows <- c(3, 7, 14)
  
  funs <- rolling_config$functions
  if (is.null(funs)) funs <- c("mean", "sd", "min", "max")
  
  for (feat in features) {
    x <- data[[feat]]
    
    for (win in windows) {
      for (fun_name in funs) {
        new_name <- paste0(feat, "_roll", win, "_", fun_name)
        
        fun <- switch(
          fun_name,
          "mean" = function(z) mean(z, na.rm = TRUE),
          "sd" = function(z) sd(z, na.rm = TRUE),
          "min" = function(z) min(z, na.rm = TRUE),
          "max" = function(z) max(z, na.rm = TRUE),
          "median" = function(z) median(z, na.rm = TRUE),
          "sum" = function(z) sum(z, na.rm = TRUE),
          function(z) mean(z, na.rm = TRUE)
        )
        
        # Calculate rolling statistic
        rolled <- numeric(length(x))
        for (i in seq_along(x)) {
          start_idx <- max(1, i - win + 1)
          window_vals <- x[start_idx:i]
          rolled[i] <- if (length(window_vals) >= win) fun(window_vals) else NA
        }
        
        data[[new_name]] <- rolled
      }
    }
  }
  
  data
}


# ==============================================================================
# STRATIFIED SPLITTING
# ==============================================================================

#' Stratified Train/Test Split
#' @keywords internal
.stratified_split <- function(y, test_ratio, seed = 42) {
  
  set.seed(seed)
  
  classes <- unique(y)
  train_idx <- c()
  test_idx <- c()
  
  for (cl in classes) {
    cl_idx <- which(y == cl)
    n_cl <- length(cl_idx)
    n_test <- max(1, round(n_cl * test_ratio))
    
    shuffled <- sample(cl_idx)
    test_idx <- c(test_idx, shuffled[seq_len(n_test)])
    train_idx <- c(train_idx, shuffled[(n_test + 1):n_cl])
  }
  
  list(train_idx = train_idx, test_idx = test_idx)
}


# ==============================================================================
# ENCODING HELPERS
# ==============================================================================

#' Target Encoding for Categorical Variables
#' @keywords internal
.target_encode <- function(data, cat_cols, target, smoothing = 1.0) {
  
  global_mean <- mean(data[[target]], na.rm = TRUE)
  encoders <- list()
  
  for (col in cat_cols) {
    # Calculate category means with smoothing
    cat_stats <- aggregate(
      data[[target]], 
      by = list(category = data[[col]]), 
      FUN = function(x) c(mean = mean(x, na.rm = TRUE), count = length(x))
    )
    
    cat_means <- cat_stats$x[, "mean"]
    cat_counts <- cat_stats$x[, "count"]
    categories <- cat_stats$category
    
    # Apply smoothing: (count * cat_mean + smoothing * global_mean) / (count + smoothing)
    smoothed_means <- (cat_counts * cat_means + smoothing * global_mean) / (cat_counts + smoothing)
    
    # Create mapping
    mapping <- setNames(smoothed_means, categories)
    encoders[[col]] <- list(mapping = mapping, global_mean = global_mean)
    
    # Apply encoding
    data[[col]] <- as.numeric(mapping[as.character(data[[col]])])
    data[[col]][is.na(data[[col]])] <- global_mean
  }
  
  list(data = data, encoders = encoders)
}


#' Frequency Encoding for Categorical Variables
#' @keywords internal
.frequency_encode <- function(data, cat_cols) {
  
  encoders <- list()
  
  for (col in cat_cols) {
    freq_table <- table(data[[col]]) / nrow(data)
    encoders[[col]] <- freq_table
    
    data[[col]] <- as.numeric(freq_table[as.character(data[[col]])])
    data[[col]][is.na(data[[col]])] <- 0
  }
  
  list(data = data, encoders = encoders)
}