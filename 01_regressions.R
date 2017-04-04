# regressions.R
# Neural Network from scratch in R
# Ilia 

set.seed(1234567)

######################################################
# 1. Linear Regression in R
#####################################################

# generate random data in which y is a noisy function of x
X <- runif(100, -5, 5)
y <- X + rnorm(100) + 3

# Plot
plot(x=X, y=y, cex = 1, col = "grey",
     main = "Explain Wages with Height", xlab = "Height", ylab = "Wages")

# Fit a model (regress weight on height)
fit <- lm(y ~ X)
print(fit)
# Coefficients:
# (Intercept)           X  
# 3.0691           0.9957

# beta-hat
fit_params <- fit$coefficients

# Draw the regression line (intercept, slope)
abline(a=fit_params[[1]], b=fit_params[[2]], col="blue") 

#####################################################
# 2. Linear Regression from Scratch
#####################################################

# Matrix of predictors (we only have one in this example)
X_mat <- as.matrix(X)
# Add column of 1s for intercept coefficient
intcpt <- rep(1, length(y))

# Combine predictors with intercept
X_mat <- cbind(intcpt, X_mat)

# OLS (closed-form solution)
beta_hat <- solve(t(X_mat) %*% X_mat) %*% t(X_mat) %*% y
print(beta_hat)
# 3.0691470, 0.9956848

# Draw the regression line
abline(a=beta_hat[[1]], b=beta_hat[[2]], col="green")

# To get y-hat:
y_hat <- X_mat %*% beta_hat
points(x=X, y=y_hat, pch = 2, col='yellow')
       
#####################################################
# 3. Linear Regression with GD
#####################################################

# Plot
plot(x=X, y=y, cex = 1, col = "green",
     main = "Explain Wages with Height", xlab = "Height", ylab = "Wages")

gradient_descent <- function(X, y, lr, epochs)
{
  X_mat <- cbind(1, X)
  # Initialise beta_hat matrix
  beta_hat <- matrix(1, nrow=ncol(X_mat))
  for (j in 1:epochs)
  {
    residual <- (X_mat %*% beta_hat) - y
    delta <- (t(X_mat) %*% residual) * (1/nrow(X_mat))
    beta_hat <- beta_hat - (lr*delta)
    # Draw the regression line
    abline(a=beta_hat[[1]], b=beta_hat[[2]], col="grey")
  }
  # Return 
  beta_hat
}

beta_hat <- gradient_descent(X, y, 0.1, 200)
print(beta_hat)
# 3.0691470, 0.9956848

# Draw the regression line
abline(a=beta_hat[[1]], b=beta_hat[[2]], col="red")

# To get y-hat:
y_hat <- X_mat %*% beta_hat
points(x=X, y=y_hat, pch = 2, col='yellow')

