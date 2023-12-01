#' @title Logistic Regression Analysis Package
#'
#' @description This package provides a comprehensive set of functions for performing logistic regression analysis. It includes functions for data preprocessing, model fitting, loss computation, bootstrap sampling, plotting logistic curves, predicting probabilities and classes, computing confusion matrices, and calculating various metrics.
#'
#' @param file A \code{string} specifying the path to the CSV file for `read_and_prepare_data`.
#' @param data A \code{data.frame} used in various functions to represent the prepared or sampled data.
#' @param model The logistic regression model used in functions like `loss_function_lr`, `plot_logistic_curve`, and others.
#' @param num_bootstraps An \code{integer} specifying the number of bootstrap samples for `bootstrap_lr`.
#' @param predictions The predicted class labels for `confusion_matrix_lr`.
#' @param true_values The actual class labels for `confusion_matrix_lr`.
#' @param cutoff_values A \code{vector} of cutoff values used for classification in `calculate_metrics`.
#' @param predictor_columns A vector of column names used as predictors in the logistic model for `bootstrap_lr`.
#' @param response_column The name of the response column in the logistic model for `bootstrap_lr`.
#' @param typeofmetric A string specifying the metric to plot in `compute_and_plot_logistic_metrics`.
#' @param plot_metric Optionally, a specific metric to plot in `compute_and_plot_logistic_metrics`.
#'
#' @return Depends on the function called, ranging from processed data.frames, logistic regression models, loss values, matrices of coefficients, plots, predicted probabilities and classes, confusion matrices, to lists of calculated metrics.
#'
#' @importFrom stats glm runif
#' @importFrom graphics plot
#' @export
#' @examples
#' # Load the necessary functions (assuming they are defined in a script or package)
#' # source("path_to_your_script.R") # Uncomment and set the path if these functions are in a script
#' # library(your_package_name)      # Uncomment and set the package name if these functions are in a package
#'
#' # Step 1: Read and preprocess the data
#' data <- read.csv(file = "expenses.csv", header = TRUE)
#' data$sex <- ifelse(data$sex == "male", 1, 0)
#' data$smoker <- ifelse(data$smoker == "yes", 1, 0)
#' data$regionnortheast <- ifelse(data$region == "northeast", 1, 0)
#' data$regionnorthwest <- ifelse(data$region == "northwest", 1, 0)
#' data$regionsoutheast <- ifelse(data$region == "southeast", 1, 0)
#' x <- as.matrix(data[, c(1:4, 7:10)])
#' X <- cbind(rep(1, nrow(x)), x)
#' y <- as.matrix(data[, 5])
#'
#' # Step 2: Fit the logistic regression model
#' fit_result <- beta_ls(predictor = X, response = y)
#'
#' # Step 3: Plot the logistic regression curve
#' plot_logistic_curve(X, fit_result$coefficient, y)
#'
#' # Step 4: Calculate and plot logistic regression metrics
#' names <- c("Prevalence", "Accuracy", "Sensitivity", "Specificity", "False Discovery Rate", "Diagnostic Odds Ratio")
#' for (metric in names) {
#'   compute_and_plot_logistic_metrics(predictor = X, response = y, typeofmetric = metric)
#' }
#'
#' # Additional analyses
#' # Bootstrap sampling
#' bootstrap_results <- bootstrap_lr(data, predictor_columns = c("sex", "smoker", "regionnortheast", "regionnorthwest", "regionsoutheast"), response_column = "expenses", num_bootstraps = 100)
#'
#' # Confusion matrix and metrics calculation
#' conf_matrix_results <- ls_confusion(predictor = X, response = y)
#' print(conf_matrix_results)
#'
#' @author Yingshan Qiu, Shakiru Oluwasanjo Oyeniran, and Sk Nafiz Rahaman
beta_ls <- function(predictor, response){
  par <- optim(beta1, loss_func_lr, predictor=predictor, response=response)$par
  loss <- loss_func_lr(par,predictor=X,response=y)
  output <- list(coefficient=par, loss=loss)
  return(output)
}

data <- read.csv(file = "expenses.csv", header = TRUE)
head(data)

data <- read.csv(file = "expenses.csv", header = TRUE)
data$sex <- ifelse(data$sex=="male",1,0)
data$smoker <- ifelse(data$smoker=="yes",1,0)
data$regionnortheast <- ifelse(data$region=="northeast",1,0)
data$regionnorthwest <- ifelse(data$region=="northwest",1,0)
data$regionsoutheast <- ifelse(data$region=="southeast",1,0)
x <- as.matrix(data[,c(1:4,7:10)])
X <- cbind(rep(1,nrow(x)),x)
y <- as.matrix(data[,5])

beta1 <- solve(t(X)%*%X)%*%t(X)%*%y

loss_func_lr <- function(beta,predictor,response){
  pi <- 1/(1+exp(-predictor%*%beta))
  loss <- sum(-response*log(pi)-(1-response)*log(1-pi))
  return(loss)
}

loss_init <- loss_func_lr(beta=beta1, predictor=X,response=y)



optim_beta <- beta_ls(predictor=X,response=y)
optim_beta

bootstrap_lr <- function(data, predictor_columns, response_column, num_bootstraps = 20) {
  # Determine the number of predictors (including intercept)
  num_predictors <- length(predictor_columns) + 1

  # Initialize a matrix to store the coefficients for each bootstrap sample
  coefficients <- matrix(NA, nrow = num_bootstraps, ncol = num_predictors)

  for (i in 1:num_bootstraps) {
    # Resample the data
    sample_indices <- sample(1:nrow(data), replace = TRUE)
    sample_data <- data[sample_indices, ]

    # Fit the logistic regression model
    formula <- as.formula(paste(response_column, "~", paste(predictor_columns, collapse = "+")))
    fitted_model <- glm(formula, data = sample_data, family = "binomial")

    # Store the coefficients
    # Use `coef` function to extract coefficients and align with matrix dimensions
    model_coefficients <- coef(fitted_model)
    coefficients[i, 1:length(model_coefficients)] <- model_coefficients
  }

  return(coefficients)
}

plot_logistic_curve <- function(X, beta_optim, y) {
  # Calculate Xβ values
  X_beta_values <- X %*% beta_optim

  # Generate a sequence for plotting
  seq_X_beta <- seq(min(X_beta_values), max(X_beta_values), length.out = 100)

  # Calculate predicted probabilities using the logistic function
  predicted_probabilities <- 1 / (1 + exp(-seq_X_beta))

  # Plot the curve
  plot(seq_X_beta, predicted_probabilities, type = "l", col = "blue",
       xlab = "Xβ", ylab = "Predicted Probability",
       main = "Fitted Logistic Curve")
  points(X_beta_values, y, col = "red")
}

# Calculate predicted probabilities
pred_prob_ls <- function(predictor, response) {
  optim_beta <- beta_ls(predictor,response)
  beta_optim <- optim_beta$coefficient
  predicted_probabilities <- 1 / (1 + exp(-predictor %*% beta_optim))
  predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)
  output <- list(prob=predicted_probabilities,class=predicted_classes)
  return(output)
}

# Calculate the confusion martrix
ls_confusion <- function(predictor, response){
  predicted_probabilities <- pred_prob_ls(predictor,response)$prob
  predicted_classes <- pred_prob_ls(predictor,response)$class

  # Create Confusion Matrix
  conf_matrix <- table(Predicted = predicted_classes, Actual = response)

  # Calculate metrics
  prevalence <- mean(response)
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
  specificity <- conf_matrix[1, 1] / sum(conf_matrix[, 1])
  false_discovery_rate <- conf_matrix[2, 1] / sum(conf_matrix[2, ])
  diagnostic_odds_ratio <- (conf_matrix[2, 2] * conf_matrix[1, 1]) / (conf_matrix[1, 2] * conf_matrix[2, 1])

  # Output
  list(ConfusionMatrix = conf_matrix,
       Prevalence = prevalence,
       Accuracy = accuracy,
       Sensitivity = sensitivity,
       Specificity = specificity,
       FalseDiscoveryRate = false_discovery_rate,
       DiagnosticOddsRatio = diagnostic_odds_ratio
  )
}

ls_confusion(predictor = X, response = y)

pred_prob_ls2 <- function(predictor, response, cutoff_value) {
  optim_beta <- beta_ls(predictor,response)
  beta_optim <- optim_beta$coefficient
  predicted_probabilities <- 1 / (1 + exp(-predictor %*% beta_optim))
  predicted_classes <- ifelse(predicted_probabilities > cutoff_value, 1, 0)
  output <- list(prob=predicted_probabilities,class=predicted_classes)
  return(output)
}

# Function to calculate metrics for a given cut-off value
calculate_metrics2 <- function(predictor, response, cutoff_value) {
  predicted_probabilities <- pred_prob_ls2(predictor, response, cutoff_value)$prob
  predicted_classes <- pred_prob_ls2(predictor, response, cutoff_value)$class

  conf_matrix <- table(Predicted = predicted_classes, Actual = response)

  # Calculate various metrics
  prevalence <- mean(y)
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
  specificity <- conf_matrix[1, 1] / sum(conf_matrix[, 1])
  false_discovery_rate <- conf_matrix[2, 1] / sum(conf_matrix[2, ])
  diagnostic_odds_ratio <- (conf_matrix[2, 2] * conf_matrix[1, 1]) / (conf_matrix[1, 2] * conf_matrix[2, 1])

  # Output
  output <- list(conf_matrix=conf_matrix,
                 prevalence=prevalence,
                 accuracy=accuracy,
                 sensitivity=sensitivity,
                 specificity=specificity,
                 false_discovery_rate=false_discovery_rate,
                 diagnostic_odds_ratio=diagnostic_odds_ratio)
  return(output)
}

compute_and_plot_logistic_metrics <- function(predictor, response, typeofmetric, plot_metric = NULL){
  cutoff_values <- seq(0.1, 0.9, by = 0.1)
  metric <- rep(0, length(cutoff_values))
  name <- c("Prevalence", "Accuracy", "Sensitivity", "Specificity", "False Discovery Rate", "Diagnostic Odds Ratio")
  if (!(typeofmetric %in% name)) {
    stop("Invalid metric name. Choose from Prevalence, Accuracy, Sensitivity, Specificity, False Discovery Rate, Diagnostic Odds Ratio.")}else{
      for (i in 1:length(cutoff_values)){
        number <- which(typeofmetric==name)+1
        metric[i] <- calculate_metrics2(predictor, response, cutoff_value=cutoff_values[i])[[number]]
      }
      plot(cutoff_values, metric, type = "b", col = "blue",
           xlab="Cut-off Value", ylab="", main=paste("Plot of",typeofmetric,"over Different Cut-off Values", sep = " "))
    }
}

for (i in c(1:6)){
  name <- c("Prevalence", "Accuracy", "Sensitivity", "Specificity", "False Discovery Rate", "Diagnostic Odds Ratio")
  compute_and_plot_logistic_metrics(predictor=X, response=y, typeofmetric=name[i], plot_metric = NULL)
}


