#' Plot Training History for easyNNR Models
#'
#' @description
#' Creates ggplot2 visualizations of training history, showing loss and metrics
#' over epochs for both training and validation sets.
#'
#' @param x An easyNNR object
#' @param ... Additional arguments passed to easy_plot()
#'
#' @return A ggplot2 object
#'
#' @export
plot.easyNNR <- function(x, ...) {
  easy_plot(x, ...)
}


#' Create Training History Plot
#'
#' @description
#' Generate publication-quality plots of training history with customizable options.
#'
#' @param object An easyNNR object
#' @param metrics Character vector of metrics to plot (default: all metrics)
#' @param smooth Logical, whether to add smoothed trend line (default: FALSE)
#' @param theme Character string: "minimal", "classic", "bw", "light" (default: "minimal")
#' @param title Character string for plot title (default: auto-generated)
#' @param subtitle Character string for plot subtitle (default: NULL)
#'
#' @return A ggplot2 object that can be further customized
#'
#' @examples
#' \dontrun{
#' # Basic plot
#' model <- easy_nn(iris, target = "Species")
#' easy_plot(model)
#' 
#' # Customized plot
#' easy_plot(model, 
#'           smooth = TRUE,
#'           theme = "classic",
#'           title = "Iris Classification Training")
#' 
#' # Plot specific metrics
#' easy_plot(model, metrics = c("loss", "accuracy"))
#' }
#'
#' @export
easy_plot <- function(object, 
                      metrics = NULL,
                      smooth = FALSE,
                      theme = "minimal",
                      title = NULL,
                      subtitle = NULL) {
  
  stopifnot(inherits(object, "easyNNR"))
  
  hist <- object$history
  
  # Filter metrics if specified
  if (!is.null(metrics)) {
    hist <- hist[hist$metric %in% metrics, ]
    if (nrow(hist) == 0) {
      stop("No data for specified metrics: ", paste(metrics, collapse = ", "))
    }
  }
  
  # Clean metric names for display
  hist$metric_clean <- gsub("_", " ", hist$metric)
  hist$metric_clean <- tools::toTitleCase(hist$metric_clean)
  
  # Clean dataset names
  hist$dataset_clean <- tools::toTitleCase(hist$dataset)
  
  # Create base plot
  p <- ggplot2::ggplot(hist, ggplot2::aes(
    x = epoch,
    y = value,
    color = dataset_clean,
    linetype = dataset_clean
  )) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::facet_wrap(~metric_clean, scales = "free_y", ncol = 2)
  
  # Add smoothed trend if requested
  if (smooth) {
    p <- p + ggplot2::geom_smooth(
      se = FALSE,
      linewidth = 0.5,
      alpha = 0.5,
      method = "loess"
    )
  }
  
  # Apply theme
  theme_fn <- switch(
    theme,
    "minimal" = ggplot2::theme_minimal,
    "classic" = ggplot2::theme_classic,
    "bw" = ggplot2::theme_bw,
    "light" = ggplot2::theme_light,
    ggplot2::theme_minimal
  )
  
  p <- p + theme_fn(base_size = 12)
  
  # Labels
  if (is.null(title)) {
    title <- paste0(
      "Training History - ",
      tools::toTitleCase(object$parameters$task)
    )
  }
  
  p <- p + ggplot2::labs(
    x = "Epoch",
    y = "Value",
    color = "Dataset",
    linetype = "Dataset",
    title = title,
    subtitle = subtitle
  )
  
  # Color scheme
  p <- p + ggplot2::scale_color_manual(
    values = c("Training" = "#1f77b4", "Validation" = "#ff7f0e")
  )
  
  # Additional theme customizations
  p <- p + ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", hjust = 0.5, size = 14),
    plot.subtitle = ggplot2::element_text(hjust = 0.5, size = 11),
    legend.position = "bottom",
    strip.background = ggplot2::element_rect(fill = "grey90", color = "grey50"),
    strip.text = ggplot2::element_text(face = "bold", size = 11),
    panel.grid.minor = ggplot2::element_blank()
  )
  
  return(p)
}


#' Plot Confusion Matrix (Classification Only)
#'
#' @description
#' Create a heatmap visualization of the confusion matrix for classification models.
#'
#' @param object An easyNNR classification model
#' @param normalize Logical, whether to show percentages instead of counts (default: TRUE)
#' @param title Character string for plot title (default: "Confusion Matrix")
#'
#' @return A ggplot2 object
#'
#' @examples
#' \dontrun{
#' model <- easy_nn(iris, target = "Species", task = "classification")
#' easy_plot_confusion(model)
#' easy_plot_confusion(model, normalize = FALSE)
#' }
#'
#' @export
easy_plot_confusion <- function(object, normalize = TRUE, title = "Confusion Matrix") {
  
  stopifnot(inherits(object, "easyNNR"))
  
  if (object$parameters$task != "classification") {
    stop("Confusion matrix plot is only available for classification tasks")
  }
  
  conf_matrix <- object$evaluation$confusion_matrix
  
  # Convert to data frame for ggplot
  conf_df <- as.data.frame(conf_matrix)
  colnames(conf_df) <- c("Actual", "Predicted", "Count")
  
  # Normalize if requested
  if (normalize) {
    totals <- aggregate(Count ~ Actual, data = conf_df, FUN = sum)
    colnames(totals) <- c("Actual", "Total")
    conf_df <- merge(conf_df, totals, by = "Actual")
    conf_df$Percentage <- conf_df$Count / conf_df$Total * 100
    fill_var <- "Percentage"
    fill_label <- "Percentage (%)"
  } else {
    fill_var <- "Count"
    fill_label <- "Count"
  }
  
  # Create heatmap
  p <- ggplot2::ggplot(
    conf_df,
    ggplot2::aes(x = Predicted, y = Actual, fill = .data[[fill_var]])
  ) +
    ggplot2::geom_tile(color = "white", linewidth = 1) +
    ggplot2::geom_text(
      ggplot2::aes(label = if (normalize) {
        sprintf("%.1f%%\n(%d)", Percentage, Count)
      } else {
        Count
      }),
      size = 4,
      color = "black"
    ) +
    ggplot2::scale_fill_gradient(
      low = "#fff7bc",
      high = "#d95f0e",
      name = fill_label
    ) +
    ggplot2::labs(
      x = "Predicted Class",
      y = "Actual Class",
      title = title
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", hjust = 0.5, size = 14),
      axis.text = ggplot2::element_text(size = 11),
      axis.title = ggplot2::element_text(face = "bold", size = 12),
      legend.position = "right",
      panel.grid = ggplot2::element_blank()
    ) +
    ggplot2::coord_fixed()
  
  return(p)
}


#' Plot Actual vs Predicted (Regression Only)
#'
#' @description
#' Create a scatter plot of actual vs predicted values for regression models,
#' with a reference line showing perfect predictions.
#'
#' @param object An easyNNR regression model
#' @param title Character string for plot title (default: "Actual vs Predicted")
#' @param show_residuals Logical, whether to color points by residual size (default: TRUE)
#'
#' @return A ggplot2 object
#'
#' @examples
#' \dontrun{
#' model <- easy_nn(mtcars, target = "mpg", task = "regression")
#' easy_plot_regression(model)
#' easy_plot_regression(model, show_residuals = FALSE)
#' }
#'
#' @export
easy_plot_regression <- function(object, 
                                  title = "Actual vs Predicted",
                                  show_residuals = TRUE) {
  
  stopifnot(inherits(object, "easyNNR"))
  
  if (object$parameters$task != "regression") {
    stop("Regression plot is only available for regression tasks")
  }
  
  pred_df <- object$predictions
  
  # Create base plot
  if (show_residuals) {
    p <- ggplot2::ggplot(pred_df, ggplot2::aes(
      x = actual,
      y = predicted,
      color = abs(residual)
    ))
    color_label <- "Abs. Residual"
  } else {
    p <- ggplot2::ggplot(pred_df, ggplot2::aes(x = actual, y = predicted))
    color_label <- NULL
  }
  
  # Add reference line (perfect predictions)
  range_vals <- range(c(pred_df$actual, pred_df$predicted))
  
  p <- p +
    ggplot2::geom_abline(
      intercept = 0,
      slope = 1,
      linetype = "dashed",
      color = "gray50",
      linewidth = 1
    ) +
    ggplot2::geom_point(size = 3, alpha = 0.6) +
    ggplot2::labs(
      x = "Actual Values",
      y = "Predicted Values",
      title = title,
      subtitle = paste0(
        "R² = ", round(object$evaluation$r_squared, 3),
        " | RMSE = ", round(object$evaluation$rmse, 3)
      ),
      color = color_label
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", hjust = 0.5, size = 14),
      plot.subtitle = ggplot2::element_text(hjust = 0.5, size = 11),
      legend.position = "right"
    )
  
  # Color scale for residuals
  if (show_residuals) {
    p <- p + ggplot2::scale_color_gradient(
      low = "#2c7bb6",
      high = "#d7191c",
      name = "Abs. Residual"
    )
  } else {
    p <- p + ggplot2::geom_point(color = "#1f77b4")
  }
  
  return(p)
}


#' Plot Residuals (Regression Only)
#'
#' @description
#' Create a residual plot for regression models to check for patterns
#' in prediction errors.
#'
#' @param object An easyNNR regression model
#' @param title Character string for plot title (default: "Residual Plot")
#'
#' @return A ggplot2 object
#'
#' @examples
#' \dontrun{
#' model <- easy_nn(mtcars, target = "mpg", task = "regression")
#' easy_plot_residuals(model)
#' }
#'
#' @export
easy_plot_residuals <- function(object, title = "Residual Plot") {
  
  stopifnot(inherits(object, "easyNNR"))
  
  if (object$parameters$task != "regression") {
    stop("Residual plot is only available for regression tasks")
  }
  
  pred_df <- object$predictions
  
  p <- ggplot2::ggplot(pred_df, ggplot2::aes(x = predicted, y = residual)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
    ggplot2::geom_point(size = 3, alpha = 0.6, color = "#1f77b4") +
    ggplot2::geom_smooth(se = TRUE, color = "darkblue", linewidth = 0.8) +
    ggplot2::labs(
      x = "Predicted Values",
      y = "Residuals",
      title = title,
      subtitle = "Residuals should be randomly scattered around zero"
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", hjust = 0.5, size = 14),
      plot.subtitle = ggplot2::element_text(hjust = 0.5, size = 10, color = "gray50")
    )
  
  return(p)
}


#' Plot Feature Importance
#'
#' @description
#' Create a bar plot showing feature importance based on permutation importance.
#'
#' @param object An easyNNR model
#' @param X_test Test feature matrix (optional, uses stored test data if available)
#' @param y_test Test target vector (optional)
#' @param top_n Number of top features to display (default: 20)
#' @param title Plot title
#'
#' @return A ggplot2 object
#'
#' @export
easy_plot_importance <- function(object, X_test = NULL, y_test = NULL, 
                                  top_n = 20, title = "Feature Importance") {
  
  stopifnot(inherits(object, "easyNNR"))
  
  # Get feature names from recipe
  feature_names <- colnames(recipes::bake(object$recipe, new_data = object$training_data))
  feature_names <- setdiff(feature_names, object$target)
  
  # Use stored predictions data to approximate importance
  if (is.null(X_test) || is.null(y_test)) {
    message("Computing feature importance from training data...")
    
    baked <- recipes::bake(object$recipe, new_data = object$training_data)
    X_test <- as.matrix(baked[, feature_names, drop = FALSE])
    
    if (object$parameters$task == "regression") {
      y_test <- as.numeric(baked[[object$target]])
    } else {
      y_test <- as.integer(factor(baked[[object$target]], 
                                   levels = object$parameters$class_levels)) - 1L
    }
  }
  
  # Calculate permutation importance
  importance <- .permutation_importance(
    object$model, 
    X_test, 
    y_test, 
    feature_names
  )
  
  # Select top N features
  importance <- head(importance, top_n)
  importance$feature <- factor(importance$feature, levels = rev(importance$feature))
  
  # Create plot
  p <- ggplot2::ggplot(importance, ggplot2::aes(x = feature, y = importance)) +
    ggplot2::geom_bar(stat = "identity", fill = "#1f77b4", alpha = 0.8) +
    ggplot2::coord_flip() +
    ggplot2::labs(
      x = "Feature",
      y = "Importance (%)",
      title = title,
      subtitle = "Based on permutation importance"
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", hjust = 0.5, size = 14),
      plot.subtitle = ggplot2::element_text(hjust = 0.5, size = 10, color = "gray50")
    )
  
  return(p)
}


#' Plot Class Distribution (Classification Only)
#'
#' @description
#' Create a bar plot showing the distribution of classes in training data
#' and predictions.
#'
#' @param object An easyNNR classification model
#' @param title Plot title
#'
#' @return A ggplot2 object
#'
#' @export
easy_plot_class_distribution <- function(object, title = "Class Distribution") {
  
  stopifnot(inherits(object, "easyNNR"))
  
  if (object$parameters$task != "classification") {
    stop("Class distribution plot is only available for classification tasks")
  }
  
  pred_df <- object$predictions
  
  # Get actual and predicted distributions
  actual_dist <- data.frame(
    class = names(table(pred_df$actual)),
    count = as.numeric(table(pred_df$actual)),
    type = "Actual"
  )
  
  pred_dist <- data.frame(
    class = names(table(pred_df$predicted)),
    count = as.numeric(table(pred_df$predicted)),
    type = "Predicted"
  )
  
  plot_data <- rbind(actual_dist, pred_dist)
  
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = class, y = count, fill = type)) +
    ggplot2::geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    ggplot2::scale_fill_manual(values = c("Actual" = "#1f77b4", "Predicted" = "#ff7f0e")) +
    ggplot2::labs(
      x = "Class",
      y = "Count",
      fill = "",
      title = title
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", hjust = 0.5, size = 14),
      legend.position = "bottom"
    )
  
  return(p)
}


#' Plot Residual Distribution (Regression Only)
#'
#' @description
#' Create a histogram and density plot of residuals for regression models.
#'
#' @param object An easyNNR regression model
#' @param title Plot title
#'
#' @return A ggplot2 object
#'
#' @export
easy_plot_residual_dist <- function(object, title = "Residual Distribution") {
  
  stopifnot(inherits(object, "easyNNR"))
  
  if (object$parameters$task != "regression") {
    stop("Residual distribution plot is only available for regression tasks")
  }
  
  pred_df <- object$predictions
  
  p <- ggplot2::ggplot(pred_df, ggplot2::aes(x = residual)) +
    ggplot2::geom_histogram(
      ggplot2::aes(y = ggplot2::after_stat(density)),
      bins = 30,
      fill = "#1f77b4",
      alpha = 0.6,
      color = "white"
    ) +
    ggplot2::geom_density(color = "darkblue", linewidth = 1) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
    ggplot2::labs(
      x = "Residual",
      y = "Density",
      title = title,
      subtitle = paste0(
        "Mean: ", round(mean(pred_df$residual), 3),
        " | SD: ", round(sd(pred_df$residual), 3)
      )
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", hjust = 0.5, size = 14),
      plot.subtitle = ggplot2::element_text(hjust = 0.5, size = 11)
    )
  
  return(p)
}


#' Create a Dashboard of Plots
#'
#' @description
#' Generate a multi-panel dashboard with key visualizations.
#'
#' @param object An easyNNR model
#'
#' @return A combined ggplot2 object (uses patchwork if available)
#'
#' @export
easy_dashboard <- function(object) {
  
  stopifnot(inherits(object, "easyNNR"))
  
  plots <- list()
  
  # Training history
  plots$history <- easy_plot(object, title = "Training History")
  
  if (object$parameters$task == "classification") {
    plots$confusion <- easy_plot_confusion(object)
    plots$class_dist <- easy_plot_class_distribution(object)
  } else {
    plots$regression <- easy_plot_regression(object)
    plots$residuals <- easy_plot_residuals(object)
    plots$residual_dist <- easy_plot_residual_dist(object)
  }
  
  # Try to combine with patchwork if available
  if (requireNamespace("patchwork", quietly = TRUE)) {
    combined <- patchwork::wrap_plots(plots, ncol = 2)
    return(combined)
  } else {
    message("Install 'patchwork' package for combined dashboard view")
    return(plots)
  }
}