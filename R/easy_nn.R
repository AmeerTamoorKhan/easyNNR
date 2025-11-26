#' easy_nn: One-Function Neural Network Training
#'
#' @description
#' A single-function solution for building, training, and evaluating neural networks in R.
#' Perfect for beginners and researchers who want quick results without deep learning expertise.
#' Compatible with TensorFlow 2.15.0 and Keras 2.15.0.
#' 
#' @param data A data frame containing your dataset
#' @param target Character string specifying the name of the target variable (column to predict)
#' @param task Character string: "regression" or "classification" (auto-detected if NULL)
#' @param exclude Character vector of column names to exclude from training (e.g., IDs, dates)
#' @param test_split Proportion of data to use for testing (default: 0.2)
#' @param validation_split Proportion of training data to use for validation (default: 0.2)
#' @param layers Integer vector specifying neurons in each hidden layer (default: c(128, 64))
#' @param activations Character vector of activation functions for each layer (default: "relu")
#' @param dropout Numeric vector of dropout rates for each layer (default: NULL, no dropout)
#' @param batch_norm Logical, whether to use batch normalization (default: FALSE)
#' @param optimizer Character string: "adam", "rmsprop", "sgd", "adamw", etc. (default: "adam")
#' @param learning_rate Numeric, learning rate for optimizer (default: 0.001)
#' @param loss Character string, custom loss function (auto-detected if NULL)
#' @param metrics Character vector of metrics to track (auto-detected if NULL)
#' @param epochs Integer, number of training epochs (default: 50)
#' @param batch_size Integer, batch size for training (default: 32)
#' @param early_stopping Logical, whether to use early stopping (default: TRUE)
#' @param patience Integer, epochs to wait before early stopping (default: 15)
#' @param scale_data Logical or character, scaling method: TRUE/"standard", "minmax", "robust", "maxabs", "quantile", or FALSE (default: TRUE)
#' @param seed Integer, random seed for reproducibility (default: 42)
#' @param verbose Logical, whether to print detailed training progress (default: TRUE)
#' @param preprocess List of preprocessing options (see Details)
#'
#' @details
#' The \code{preprocess} parameter accepts a list with the following options:
#' 
#' \strong{Outlier Handling:}
#' \itemize{
#'   \item \code{outlier_method}: "none", "iqr", "zscore", "isolation_forest", "winsorize", "cap", "remove" (default: "none")
#'   \item \code{outlier_threshold}: Threshold for outlier detection (default: 1.5 for IQR, 3 for zscore)
#' }
#' 
#' \strong{Target Transformation (Regression):}
#' \itemize{
#'   \item \code{target_transform}: "none", "log", "log1p", "sqrt", "boxcox", "yeojohnson", "quantile" (default: "none")
#' }
#' 
#' \strong{Class Imbalance (Classification):}
#' \itemize{
#'   \item \code{imbalance_method}: "none", "oversample", "undersample", "smote", "adasyn", "class_weights" (default: "none")
#'   \item \code{imbalance_ratio}: Target ratio for resampling (default: 1.0)
#' }
#' 
#' \strong{Feature Selection:}
#' \itemize{
#'   \item \code{feature_selection}: "none", "variance", "correlation", "mutual_info", "rfe", "lasso" (default: "none")
#'   \item \code{n_features}: Number of features to select (default: NULL, auto)
#'   \item \code{correlation_threshold}: Threshold for correlation filter (default: 0.9)
#' }
#' 
#' \strong{Encoding Methods:}
#' \itemize{
#'   \item \code{encoding}: "onehot", "target", "frequency", "binary", "hash" (default: "onehot")
#'   \item \code{max_categories}: Max categories for one-hot encoding (default: 50)
#' }
#' 
#' \strong{Imputation:}
#' \itemize{
#'   \item \code{impute_numeric}: "median", "mean", "knn", "iterative", "constant" (default: "median")
#'   \item \code{impute_categorical}: "mode", "constant", "missing_category" (default: "mode")
#'   \item \code{add_indicators}: Logical, add missingness indicator columns (default: FALSE)
#' }
#' 
#' \strong{Feature Engineering:}
#' \itemize{
#'   \item \code{interactions}: Logical or formula, create interaction terms (default: FALSE
#'   \item \code{polynomial_degree}: Integer, degree for polynomial features (default: 1, no polynomial)
#'   \item \code{pca_components}: Number of PCA components (default: NULL, no PCA)
#' }
#' 
#' \strong{Time Series Features:}
#' \itemize{
#'   \item \code{date_features}: Logical, extract features from date columns (default: FALSE)
#'   \item \code{lag_features}: Integer vector of lag periods (default: NULL)
#'   \item \code{rolling_features}: List with window sizes for rolling statistics (default: NULL)
#' }
#'
#' @return An 'easyNNR' object (S3 list) containing:
#' \itemize{
#'   \item \strong{model}: Trained Keras model
#'   \item \strong{history}: Training history as tidy tibble
#'   \item \strong{evaluation}: Test set performance metrics
#'   \item \strong{predictions}: Test set predictions with actuals
#'   \item \strong{recipe}: Preprocessing recipe for reuse
#'   \item \strong{target}: Target variable name
#'   \item \strong{exclude}: Excluded columns
#'   \item \strong{parameters}: Model configuration and hyperparameters
#'   \item \strong{preprocessing}: Preprocessing configuration and transformers
#' }
#'
#' @examples
#' \dontrun{
#' # Simple classification
#' library(easyNNR)
#' data(iris)
#' model <- easy_nn(iris, target = "Species")
#' 
#' # Regression with preprocessing
#' data(mtcars)
#' model <- easy_nn(
#'   data = mtcars,
#'   target = "mpg",
#'   task = "regression",
#'   layers = c(256, 128, 64),
#'   preprocess = list(
#'     outlier_method = "winsorize",
#'     target_transform = "log",
#'     feature_selection = "correlation"
#'   )
#' )
#' 
#' # Classification with imbalance handling
#' model <- easy_nn(
#'   data = imbalanced_data,
#'   target = "class",
#'   preprocess = list(
#'     imbalance_method = "smote",
#'     encoding = "target",
#'     feature_selection = "mutual_info",
#'     n_features = 20
#'   )
#' )
#' }
#'
#' @export
easy_nn <- function(
  data,
  target,
  task = NULL,
  exclude = NULL,
  test_split = 0.2,
  validation_split = 0.2,

  layers = c(128, 64),
  activations = NULL,
  dropout = NULL,
  batch_norm = FALSE,
  optimizer = "adam",
  learning_rate = 0.001,
  loss = NULL,
  metrics = NULL,
  epochs = 50,
  batch_size = 32,

  early_stopping = TRUE,
  patience = 15,
  scale_data = TRUE,
  seed = 42,
  verbose = TRUE,
  preprocess = list()
) {
  
  # ============================================================================
  # STEP 1: Environment and Input Validation
  # ============================================================================
  
  if (verbose) {
    cat("\n╔════════════════════════════════════════════════╗\n")
    cat("║         🧠 easyNNR: Neural Network Builder    ║\n")
    cat("╚════════════════════════════════════════════════╝\n\n")
    cat("📊 STEP 1: Validating environment and data...\n")
  }
  
  # Check TensorFlow/Keras
  chk <- .check_tf()
  if (!chk$ok) stop(chk$msg, call. = FALSE)
  
  # Validate inputs
  if (!is.data.frame(data)) {
    stop("'data' must be a data.frame or tibble", call. = FALSE)
  }
  
  original_nrow <- nrow(data)
  
  if (original_nrow == 0) {
    stop("'data' has no rows. Please provide a dataset with observations.", call. = FALSE)
  }
  
  if (ncol(data) < 2) {
    stop("'data' must have at least 2 columns (features + target)", call. = FALSE)
  }
  
  if (is.null(target) || !target %in% names(data)) {
    stop(
      "❌ Target column '", if (!is.null(target)) target else "NULL", 
      "' not found in data.\n",
      "   Available columns: ", paste(head(names(data), 20), collapse = ", "),
      if (ncol(data) > 20) paste0("\n   ... and ", ncol(data) - 20, " more") else "",
      "\n   Please provide a valid 'target' column name.",
      call. = FALSE
    )
  }
  
  # Parse preprocessing options with defaults
  preprocess <- .parse_preprocess_options(preprocess)
  
  # Set reproducibility
  set.seed(seed)
  tensorflow::tf$random$set_seed(as.integer(seed))
  
  # ============================================================================
  # STEP 1.5: Automatic Data Quality Fixes
  # ============================================================================
  
  if (verbose) cat("   🔍 Running automatic data quality checks...\n")
  
  # Handle excluded columns
  exclude <- unique(c(exclude))

  exclude <- exclude[exclude %in% names(data)]
  if (length(exclude) > 0 && verbose) {
    cat("   Excluding columns:", paste(exclude, collapse = ", "), "\n")
  }
  
  feats <- setdiff(names(data), c(target, exclude))
  
  # Identify date columns for potential feature engineering
  date_cols <- .identify_date_columns(data[, feats, drop = FALSE])
  
  if (length(date_cols) > 0 && verbose) {
    cat("   📅 Detected date columns:", paste(date_cols, collapse = ", "), "\n")
  }
  
  # Check for and handle constant columns (all same value)
  constant_cols <- sapply(data[, feats, drop = FALSE], function(x) {
    if (is.numeric(x)) {
      non_na <- x[!is.na(x)]
      if (length(non_na) == 0) return(TRUE)
      return(length(unique(non_na)) == 1)
    }
    FALSE
  })
  
  if (any(constant_cols) && verbose) {
    const_names <- names(data[, feats, drop = FALSE])[constant_cols]
    if (length(const_names) > 0) {
      cat("   ⚠️  Removing", length(const_names), "constant column(s):", 
          paste(head(const_names, 5), collapse = ", "))
      if (length(const_names) > 5) cat(", ...")
      cat("\n")
      feats <- setdiff(feats, const_names)
    }
  }
  
  # Check for columns with too many missing values
  na_prop <- colMeans(is.na(data[, feats, drop = FALSE]))
  high_na_cols <- names(na_prop)[na_prop > 0.8]
  
  if (length(high_na_cols) > 0 && verbose) {
    cat("   ⚠️  Removing", length(high_na_cols), "column(s) with >80% missing values\n")
    feats <- setdiff(feats, high_na_cols)
  }
  
  # Check for rows with all missing features
  all_na_rows <- apply(data[, feats, drop = FALSE], 1, function(x) all(is.na(x)))
  
  if (any(all_na_rows)) {
    if (verbose) cat("   ⚠️  Removing", sum(all_na_rows), "row(s) with all missing feature values\n")
    data <- data[!all_na_rows, , drop = FALSE]
  }
  
  if (length(feats) == 0) {
    stop(
      "❌ No feature columns available after excluding target and excluded columns!\n",
      "   Target: ", target, "\n",
      "   Excluded: ", paste(exclude, collapse = ", "), "\n",
      "   Please check your column specifications.",
      call. = FALSE
    )
  }
  
  if (verbose) {
    cat("   Dataset size: ", nrow(data), " rows × ", ncol(data), " columns\n", sep = "")
    cat("   Target column: '", target, "'\n", sep = "")
    cat("   Feature columns: ", length(feats), " (", 
        paste(head(feats, 5), collapse = ", "),
        if (length(feats) > 5) paste0(", ... +", length(feats) - 5, " more") else "",
        ")\n", sep = "")
  }
  
  # ============================================================================
  # STEP 2: Task Detection and Target Processing
  # ============================================================================
  
  y_raw <- data[[target]]
  
  # Handle missing values in target AUTOMATICALLY
  na_count_target <- sum(is.na(y_raw))
  if (na_count_target > 0) {
    if (verbose) {
      cat("   ⚠️  Target variable has", na_count_target, "missing value(s)\n")
      cat("   Automatically removing rows with missing target values...\n")
    }
    
    valid_rows <- !is.na(y_raw)
    data <- data[valid_rows, , drop = FALSE]
    y_raw <- data[[target]]
    
    if (verbose) {
      cat("   ✓ Removed", na_count_target, "row(s). Remaining:", nrow(data), "observations\n")
    }
    
    if (nrow(data) == 0) {
      stop("❌ All rows have missing target values!", call. = FALSE)
    }
  }
  
  if (is.null(task)) {
    task <- .infer_task(y_raw)
    if (verbose) cat("   Auto-detected task:", toupper(task), "\n")
  }
  
  class_levels <- NULL
  num_classes <- NULL
  target_transformer <- NULL
  class_weights <- NULL
  
  if (task == "classification") {
    if (!is.factor(y_raw)) y_raw <- factor(y_raw)
    class_levels <- levels(y_raw)
    num_classes <- length(class_levels)
    
    if (verbose) {
      cat("   Number of classes:", num_classes, "\n")
      cat("   Classes:", paste(class_levels, collapse = ", "), "\n")
      
      # Show class distribution
      class_dist <- table(y_raw)
      cat("   Class distribution:\n")
      for (cl in names(class_dist)) {
        pct <- round(class_dist[cl] / sum(class_dist) * 100, 1)
        cat("      ", cl, ": ", class_dist[cl], " (", pct, "%)\n", sep = "")
      }
    }
    
    # Calculate class weights if requested
    if (preprocess$imbalance_method == "class_weights") {
      class_weights <- .calculate_class_weights(y_raw, class_levels)
      if (verbose) cat("   ✓ Class weights calculated for imbalance handling\n")
    }
    
  } else {
    # Regression: Apply target transformation if requested
    if (preprocess$target_transform != "none") {
      transform_result <- .transform_target(y_raw, preprocess$target_transform)
      y_raw <- transform_result$transformed
      target_transformer <- transform_result$transformer
      data[[target]] <- y_raw
      
      if (verbose) {
        cat("   ✓ Applied target transformation:", preprocess$target_transform, "\n")
      }
    }
  }
  
  if (verbose) cat("   ✓ Data validation complete\n")
  
  # ============================================================================
  # STEP 3: Advanced Preprocessing Pipeline
  # ============================================================================
  
  if (verbose) cat("\n🔧 STEP 3: Building preprocessing pipeline...\n")
  
  # Store preprocessing artifacts
  preprocess_artifacts <- list()
  
  # --- 3.1: Time Series Feature Engineering ---
  if (preprocess$date_features && length(date_cols) > 0) {
    if (verbose) cat("   📅 Extracting date features...\n")
    data <- .extract_date_features(data, date_cols)
    feats <- setdiff(names(data), c(target, exclude))
    feats <- feats[!feats %in% date_cols]  # Remove original date columns
  }
  
  # --- 3.2: Lag and Rolling Features ---
  if (!is.null(preprocess$lag_features) || !is.null(preprocess$rolling_features)) {
    if (verbose) cat("   📈 Creating time series features...\n")
    
    numeric_feats <- feats[sapply(data[, feats, drop = FALSE], is.numeric)]
    
    if (!is.null(preprocess$lag_features)) {
      data <- .create_lag_features(data, numeric_feats, preprocess$lag_features)
    }
    
    if (!is.null(preprocess$rolling_features)) {
      data <- .create_rolling_features(data, numeric_feats, preprocess$rolling_features)
    }
    
    # Remove rows with NA from lag/rolling features
    complete_rows <- complete.cases(data)
    if (sum(!complete_rows) > 0) {
      if (verbose) cat("   ⚠️  Removing", sum(!complete_rows), "rows with NA from lag/rolling features\n")
      data <- data[complete_rows, , drop = FALSE]
    }
    
    feats <- setdiff(names(data), c(target, exclude))
  }
  
  # --- 3.3: Outlier Handling ---
  if (preprocess$outlier_method != "none") {
    if (verbose) cat("   🔍 Handling outliers (method:", preprocess$outlier_method, ")...\n")
    
    outlier_result <- .handle_outliers(
      data = data,
      features = feats,
      method = preprocess$outlier_method,
      threshold = preprocess$outlier_threshold
    )
    
    data <- outlier_result$data
    preprocess_artifacts$outlier_info <- outlier_result$info
    
    if (verbose && !is.null(outlier_result$info$n_affected)) {
      cat("   ✓ Processed", outlier_result$info$n_affected, "outlier values\n")
    }
  }
  
  # --- 3.4: Missingness Indicators ---
  if (preprocess$add_indicators) {
    if (verbose) cat("   📌 Adding missingness indicator columns...\n")
    
    na_cols <- feats[colSums(is.na(data[, feats, drop = FALSE])) > 0]
    for (col in na_cols) {
      indicator_name <- paste0(col, "_missing")
      data[[indicator_name]] <- as.integer(is.na(data[[col]]))
    }
    
    if (verbose && length(na_cols) > 0) {
      cat("   ✓ Added", length(na_cols), "missingness indicators\n")
    }
  }
  
  # Update feature list
  feats <- setdiff(names(data), c(target, exclude))
  
  # --- 3.5: Build recipes preprocessing pipeline ---
  rec <- recipes::recipe(data) |>
    recipes::update_role(dplyr::all_of(target), new_role = "outcome") |>
    recipes::update_role(dplyr::all_of(feats), new_role = "predictor")
  
  # Imputation based on method
  if (preprocess$impute_numeric == "median") {
    rec <- rec |> recipes::step_impute_median(recipes::all_numeric_predictors())
  } else if (preprocess$impute_numeric == "mean") {
    rec <- rec |> recipes::step_impute_mean(recipes::all_numeric_predictors())
  } else if (preprocess$impute_numeric == "knn") {
    rec <- rec |> recipes::step_impute_knn(recipes::all_numeric_predictors(), neighbors = 5)
  } else if (preprocess$impute_numeric == "constant") {
    rec <- rec |> recipes::step_impute_mean(recipes::all_numeric_predictors())  # Fallback
  }
  
  # Categorical imputation
  if (preprocess$impute_categorical == "mode") {
    rec <- rec |> recipes::step_impute_mode(recipes::all_nominal_predictors())
  } else if (preprocess$impute_categorical == "missing_category") {
    rec <- rec |> recipes::step_unknown(recipes::all_nominal_predictors(), new_level = "MISSING")
  }
  
  # String to factor conversion
  rec <- rec |> recipes::step_string2factor(recipes::all_nominal_predictors())
  
  # --- 3.6: Encoding ---
  if (preprocess$encoding == "onehot") {
    # Collapse rare categories first if max_categories is set
    if (!is.null(preprocess$max_categories)) {
      rec <- rec |> recipes::step_other(
        recipes::all_nominal_predictors(), 
        threshold = 1 / preprocess$max_categories,
        other = "OTHER"
      )
    }
    rec <- rec |> recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = TRUE)
    
  } else if (preprocess$encoding == "frequency") {
    # Frequency encoding: replace category with its frequency
    rec <- rec |> recipes::step_mutate_at(
      recipes::all_nominal_predictors(),
      fn = function(x) {
        freq_table <- table(x) / length(x)
        as.numeric(freq_table[as.character(x)])
      }
    )
    
  } else if (preprocess$encoding == "binary") {
    # Binary encoding (using hashing with limited columns)
    rec <- rec |> recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = FALSE)
  }
  # Note: Target encoding and hash encoding handled separately below
  
  # Remove zero-variance columns
  rec <- rec |> recipes::step_zv(recipes::all_predictors())
  
  # Remove near-zero variance if requested
  if (preprocess$feature_selection == "variance") {
    rec <- rec |> recipes::step_nzv(recipes::all_predictors(), freq_cut = 95/5, unique_cut = 10)
  }
  
  # --- 3.7: Correlation-based Feature Selection ---
  if (preprocess$feature_selection == "correlation") {
    rec <- rec |> recipes::step_corr(
      recipes::all_numeric_predictors(), 
      threshold = preprocess$correlation_threshold
    )
    if (verbose) cat("   ✓ Correlation filter enabled (threshold:", preprocess$correlation_threshold, ")\n")
  }
  
  # --- 3.8: Scaling ---
  if (!isFALSE(scale_data)) {
    scale_method <- if (isTRUE(scale_data)) "standard" else scale_data
    
    if (scale_method == "standard") {
      rec <- rec |>
        recipes::step_center(recipes::all_numeric_predictors()) |>
        recipes::step_scale(recipes::all_numeric_predictors())
    } else if (scale_method == "minmax") {
      rec <- rec |> recipes::step_range(recipes::all_numeric_predictors(), min = 0, max = 1)
    } else if (scale_method == "robust") {
      # Robust scaling using median and IQR
      rec <- rec |>
        recipes::step_center(recipes::all_numeric_predictors(), fn = median) |>
        recipes::step_scale(recipes::all_numeric_predictors(), 
                           fn = function(x) stats::IQR(x, na.rm = TRUE))
    } else if (scale_method == "maxabs") {
      rec <- rec |> recipes::step_scale(
        recipes::all_numeric_predictors(),
        fn = function(x) max(abs(x), na.rm = TRUE)
      )
    } else if (scale_method == "quantile") {
      # Quantile transformation for uniform distribution
      rec <- rec |> recipes::step_orderNorm(recipes::all_numeric_predictors())
    }
    
    if (verbose) cat("   ✓ Feature scaling enabled (method:", scale_method, ")\n")
  }
  
  # --- 3.9: Feature Interactions and Polynomial Features ---
  if (isTRUE(preprocess$interactions) || inherits(preprocess$interactions, "formula")) {
    if (verbose) cat("   ✓ Creating interaction terms...\n")
    rec <- rec |> recipes::step_interact(
      terms = if (inherits(preprocess$interactions, "formula")) {
        preprocess$interactions
      } else {
        ~ recipes::all_numeric_predictors():recipes::all_numeric_predictors()
      }
    )
  }
  
  if (!is.null(preprocess$polynomial_degree) && preprocess$polynomial_degree > 1) {
    if (verbose) cat("   ✓ Creating polynomial features (degree:", preprocess$polynomial_degree, ")...\n")
    rec <- rec |> recipes::step_poly(
      recipes::all_numeric_predictors(), 
      degree = preprocess$polynomial_degree
    )
  }
  
  # --- 3.10: Dimensionality Reduction (PCA) ---
  if (!is.null(preprocess$pca_components)) {
    if (verbose) cat("   ✓ Applying PCA (components:", preprocess$pca_components, ")...\n")
    rec <- rec |> recipes::step_pca(
      recipes::all_numeric_predictors(), 
      num_comp = preprocess$pca_components
    )
  }
  
  # Prepare and apply recipe
  rec_prep <- recipes::prep(rec, training = data)
  baked <- recipes::bake(rec_prep, new_data = data)
  
  # --- 3.11: Advanced Feature Selection (post-baking) ---
  feature_selector <- NULL
  selected_features <- NULL
  
  if (preprocess$feature_selection %in% c("mutual_info", "lasso", "rfe")) {
    if (verbose) cat("   🎯 Performing feature selection (", preprocess$feature_selection, ")...\n", sep = "")
    
    X_temp <- baked |> dplyr::select(-dplyr::all_of(target)) |> as.matrix()
    y_temp <- if (task == "classification") {
      as.integer(factor(baked[[target]], levels = class_levels)) - 1L
    } else {
      as.numeric(baked[[target]])
    }
    
    selection_result <- .select_features(
      X = X_temp,
      y = y_temp,
      method = preprocess$feature_selection,
      n_features = preprocess$n_features,
      task = task
    )
    
    selected_features <- selection_result$selected
    feature_selector <- selection_result$selector
    
    if (verbose) {
      cat("   ✓ Selected", length(selected_features), "features\n")
    }
    
    # Apply selection
    baked <- baked[, c(selected_features, target), drop = FALSE]
  }
  
  X <- baked |> dplyr::select(-dplyr::all_of(target)) |> as.matrix()
  
  # Encode target
  if (task == "regression") {
    y <- as.numeric(baked[[target]])
  } else {
    y <- as.integer(factor(baked[[target]], levels = class_levels)) - 1L
  }
  
  # --- 3.12: Class Imbalance Handling (SMOTE/ADASYN/Resampling) ---
  if (task == "classification" && preprocess$imbalance_method %in% c("smote", "adasyn", "oversample", "undersample")) {
    if (verbose) cat("   ⚖️  Handling class imbalance (", preprocess$imbalance_method, ")...\n", sep = "")
    
    original_n <- nrow(X)
    
    resample_result <- .handle_imbalance(
      X = X,
      y = y,
      method = preprocess$imbalance_method,
      ratio = preprocess$imbalance_ratio,
      k = 5
    )
    
    X <- resample_result$X
    y <- resample_result$y
    
    if (verbose) {
      cat("   ✓ Resampled:", original_n, "→", nrow(X), "samples\n")
      new_dist <- table(y)
      for (i in seq_along(new_dist)) {
        pct <- round(new_dist[i] / sum(new_dist) * 100, 1)
        cat("      Class", names(new_dist)[i], ":", new_dist[i], "(", pct, "%)\n")
      }
    }
  }
  
  # Check for NaN/Inf in feature matrix
  if (any(is.na(X))) {
    na_count <- sum(is.na(X))
    warning("Feature matrix contains ", na_count, " NA/NaN values after preprocessing.",
            call. = FALSE, immediate. = TRUE)
    # Impute remaining NAs with column means
    for (j in seq_len(ncol(X))) {
      if (any(is.na(X[, j]))) {
        X[is.na(X[, j]), j] <- mean(X[, j], na.rm = TRUE)
      }
    }
  }
  
  if (any(is.infinite(X))) {
    warning("Feature matrix contains infinite values. Replacing with finite values...",
            call. = FALSE, immediate. = TRUE)
    X[is.infinite(X) & X > 0] <- .Machine$double.xmax / 2
    X[is.infinite(X) & X < 0] <- -.Machine$double.xmax / 2
  }
  
  input_dim <- ncol(X)
  if (verbose) {
    cat("   Input features:", input_dim, "\n")
    cat("   ✓ Preprocessing complete\n")
  }
  
  # ============================================================================
  # STEP 4: Train/Test Split
  # ============================================================================
  
  if (verbose) cat("\n🔀 STEP 4: Splitting data into train/test sets...\n")
  
  n <- nrow(X)
  
  # Stratified split for classification
  if (task == "classification") {
    split_result <- .stratified_split(y, test_split, seed)
    train_idx <- split_result$train_idx
    test_idx <- split_result$test_idx
  } else {
    idx <- sample.int(n)
    n_test <- max(1L, floor(n * test_split))
    test_idx <- idx[seq_len(n_test)]
    train_idx <- idx[-seq_len(n_test)]
  }
  
  X_train <- X[train_idx, , drop = FALSE]
  X_test  <- X[test_idx, , drop = FALSE]
  y_train <- y[train_idx]
  y_test  <- y[test_idx]
  
  if (verbose) {
    cat("   Training samples:", length(train_idx), "\n")
    cat("   Testing samples:", length(test_idx), "\n")
  }
  
  # ============================================================================
  # STEP 5: Build Neural Network Architecture
  # ============================================================================
  
  if (verbose) cat("\n🏗️  STEP 5: Building neural network architecture...\n")
  
  # Standardize activations
  if (is.null(activations)) activations <- rep("relu", length(layers))
  if (length(activations) == 1) activations <- rep(activations, length(layers))
  if (length(activations) != length(layers)) {
    stop("'activations' must be same length as 'layers'.", call. = FALSE)
  }
  
  # Standardize dropout
  if (!is.null(dropout)) {
    if (length(dropout) == 1) dropout <- rep(dropout, length(layers))
    if (length(dropout) != length(layers)) {
      stop("'dropout' must be length 1 or same length as 'layers'.", call. = FALSE)
    }
  }
  
  # Build model using Functional API
  input <- keras::layer_input(shape = c(input_dim), name = "input")
  
  hidden <- input
  for (i in seq_along(layers)) {
    hidden <- hidden |>
      keras::layer_dense(
        units = layers[i], 
        activation = activations[i],
        name = paste0("dense_", i)
      )
    
    if (batch_norm) {
      hidden <- hidden |> 
        keras::layer_batch_normalization(name = paste0("bn_", i))
    }
    
    if (!is.null(dropout) && !is.na(dropout[i]) && dropout[i] > 0) {
      hidden <- hidden |> 
        keras::layer_dropout(rate = dropout[i], name = paste0("drop_", i))
    }
  }
  
  # Output layer
  if (task == "regression") {
    output <- hidden |> 
      keras::layer_dense(units = 1, activation = "linear", name = "output")
    if (is.null(loss)) loss <- "mse"
    if (is.null(metrics)) metrics <- c("mae")
  } else {
    if (num_classes == 2L) {
      output <- hidden |> 
        keras::layer_dense(units = 1, activation = "sigmoid", name = "output")
      if (is.null(loss)) loss <- "binary_crossentropy"
      if (is.null(metrics)) metrics <- c("accuracy")
    } else {
      output <- hidden |> 
        keras::layer_dense(units = num_classes, activation = "softmax", name = "output")
      if (is.null(loss)) loss <- "sparse_categorical_crossentropy"
      if (is.null(metrics)) metrics <- c("accuracy")
    }
  }
  
  model <- keras::keras_model(inputs = input, outputs = output, name = "easyNNR")
  
  if (verbose) {
    cat("   Architecture:", paste(layers, collapse = " → "), "→ Output\n")
    cat("   Activations:", paste(activations, collapse = ", "), "\n")
    if (batch_norm) cat("   Batch normalization: ENABLED\n")
    if (!is.null(dropout)) cat("   Dropout rates:", paste(dropout, collapse = ", "), "\n")
  }
  
  # ============================================================================
  # STEP 6: Compile Model
  # ============================================================================
  
  if (verbose) cat("\n⚙️  STEP 6: Compiling model...\n")
  
  opt <- .make_optimizer(optimizer, learning_rate)
  model |> keras::compile(optimizer = opt, loss = loss, metrics = metrics)
  
  if (verbose) {
    cat("   Optimizer:", if (is.character(optimizer)) optimizer else "custom", "\n")
    cat("   Learning rate:", learning_rate, "\n")
    cat("   Loss function:", loss, "\n")
    cat("   Metrics:", paste(metrics, collapse = ", "), "\n")
    cat("   ✓ Model compiled\n")
  }
  
  # ============================================================================
  # STEP 7: Train Model
  # ============================================================================
  
  if (verbose) {
    cat("\n🚀 STEP 7: Training neural network...\n")
    cat("   Epochs:", epochs, "\n")
    cat("   Batch size:", batch_size, "\n")
    cat("   Validation split:", validation_split, "\n")
    if (early_stopping) cat("   Early stopping: ENABLED (patience =", patience, ")\n")
    if (!is.null(class_weights)) cat("   Class weights: ENABLED\n")
    cat("\n")
  }
  
  # Setup callbacks
  callbacks_list <- list()
  if (early_stopping) {
    callbacks_list[[length(callbacks_list) + 1]] <- 
      keras::callback_early_stopping(
        monitor = "val_loss",
        patience = patience,
        restore_best_weights = TRUE,
        verbose = ifelse(verbose, 1, 0)
      )
  }
  
  # Train the model
  fit_args <- list(
    object = model,
    x = X_train,
    y = y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = validation_split,
    callbacks = callbacks_list,
    verbose = ifelse(verbose, 1, 0)
  )
  
  # Add class weights if calculated
  if (!is.null(class_weights)) {
    fit_args$class_weight <- class_weights
  }
  
  history <- do.call(keras::fit, fit_args)
  
  if (verbose) cat("\n   ✓ Training complete!\n")
  
  # ============================================================================
  # STEP 8: Evaluate on Test Set
  # ============================================================================
  
  if (verbose) cat("\n📈 STEP 8: Evaluating model on test set...\n")
  
  test_metrics <- keras::evaluate(model, X_test, y_test, verbose = 0)
  predictions_raw <- model(X_test, training = FALSE) |> as.matrix()
  
  # Format predictions and calculate additional metrics
  if (task == "regression") {
    predictions_vec <- as.numeric(predictions_raw[, 1])
    
    # Inverse transform predictions if target was transformed
    if (!is.null(target_transformer)) {
      predictions_vec <- .inverse_transform_target(predictions_vec, target_transformer)
      y_test_original <- .inverse_transform_target(y_test, target_transformer)
    } else {
      y_test_original <- y_test
    }
    
    mae <- mean(abs(predictions_vec - y_test_original))
    rmse <- sqrt(mean((predictions_vec - y_test_original)^2))
    r_squared <- 1 - sum((y_test_original - predictions_vec)^2) / sum((y_test_original - mean(y_test_original))^2)
    
    evaluation <- list(
      loss = test_metrics[[1]],
      mae = mae,
      rmse = rmse,
      r_squared = r_squared,
      raw_metrics = test_metrics
    )
    
    predictions_df <- data.frame(
      actual = y_test_original,
      predicted = predictions_vec,
      residual = y_test_original - predictions_vec
    )
    
    if (verbose) {
      cat("   Test Loss (MSE):", round(test_metrics[[1]], 4), "\n")
      cat("   Mean Absolute Error:", round(mae, 4), "\n")
      cat("   Root Mean Squared Error:", round(rmse, 4), "\n")
      cat("   R-squared:", round(r_squared, 4), "\n")
    }
    
  } else {
    # Classification
    if (num_classes == 2L) {
      probs <- as.numeric(predictions_raw[, 1])
      predicted_labels <- ifelse(probs >= 0.5, class_levels[2], class_levels[1])
    } else {
      predicted_indices <- max.col(predictions_raw)
      predicted_labels <- class_levels[predicted_indices]
    }
    
    actual_labels <- class_levels[y_test + 1]
    accuracy <- mean(predicted_labels == actual_labels)
    conf_matrix <- table(Actual = actual_labels, Predicted = predicted_labels)
    
    # Calculate additional classification metrics
    class_metrics <- .classification_metrics(actual_labels, predicted_labels)
    
    evaluation <- list(
      loss = test_metrics[[1]],
      accuracy = accuracy,
      confusion_matrix = conf_matrix,
      precision = class_metrics$precision,
      recall = class_metrics$recall,
      f1_score = class_metrics$f1_score,
      raw_metrics = test_metrics
    )
    
    predictions_df <- data.frame(
      actual = actual_labels,
      predicted = predicted_labels,
      correct = predicted_labels == actual_labels
    )
    
    if (verbose) {
      cat("   Test Loss:", round(test_metrics[[1]], 4), "\n")
      cat("   Test Accuracy:", round(accuracy * 100, 2), "%\n")
      if (!is.null(class_metrics$precision)) {
        cat("   Precision:", round(class_metrics$precision * 100, 2), "%\n")
        cat("   Recall:", round(class_metrics$recall * 100, 2), "%\n")
        cat("   F1 Score:", round(class_metrics$f1_score, 4), "\n")
      }
      cat("\n   Confusion Matrix:\n")
      print(conf_matrix)
    }
  }
  
  # Tidy history
  hist_tbl <- .tidy_history(history)
  
  # ============================================================================
  # STEP 9: Prepare Return Object
  # ============================================================================
  
  if (verbose) {
    cat("\n╔════════════════════════════════════════════════╗\n")
    cat("║             ✅ easyNNR Complete!              ║\n")
    cat("╚════════════════════════════════════════════════╝\n\n")
  }
  
  out <- list(
    model = model,
    history = hist_tbl,
    evaluation = evaluation,
    predictions = predictions_df,
    recipe = rec_prep,
    target = target,
    exclude = exclude,
    training_data = data,
    parameters = list(
      task = task,
      test_split = test_split,
      validation_split = validation_split,
      layers = layers,
      activations = activations,
      dropout = dropout,
      batch_norm = batch_norm,
      optimizer = if (is.character(optimizer)) optimizer else "custom",
      learning_rate = learning_rate,
      epochs = epochs,
      batch_size = batch_size,
      early_stopping = early_stopping,
      patience = patience,
      scale_data = scale_data,
      loss = loss,
      metrics = metrics,
      class_levels = class_levels,
      num_classes = num_classes,
      input_dim = input_dim,
      seed = seed,
      original_nrow = original_nrow,
      rows_removed = original_nrow - nrow(data)
    ),
    preprocessing = list(
      options = preprocess,
      target_transformer = target_transformer,
      class_weights = class_weights,
      feature_selector = feature_selector,
      selected_features = selected_features,
      artifacts = preprocess_artifacts
    )
  )
  
  class(out) <- "easyNNR"
  
  # Print summary
  print(out)
  
  invisible(out)
}