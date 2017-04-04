# logitregressions.R
# Neural Network from scratch in R
# Ilia 03.04.2017

#####################################################
# 1. Logistic Regression in R
#####################################################

# Two possible outcomes -> binomial
data_df <- as.data.frame(iris)
idx <- data_df$Species %in% c("virginica", "versicolor")
data_df <- data_df[idx,]
y <- ifelse(data_df$Species=="virginica",1,0)

# For faster convergence let's rescale X
X <- data_df[c(1:4)]
X <- as.matrix(X/max(X))

# Fit model
model <- glm(y ~ X, family=binomial(link='logit'))

# Params
print(model)
# Coefficients:
# (Intercept)  XSepal.Length   XSepal.Width  XPetal.Length   XPetal.Width  
# -42.64         -19.48         -52.78          74.49         144.46  
summary(model)

#####################################################
# 2. Logistic Regression with GD
#####################################################

# Calculate activation function (sigmoid for logit)
sigmoid <- function(z){1.0/(1.0+exp(-z))}

# Calculate log-likelihood (easier to max than likelihood)
log_likelihood <- function(X, y, beta_hat)
{
  scores <- X %*% beta_hat
  ll <- (y %*% scores) - log(1+exp(scores))
  sum(ll)
}

logistic_reg <- function(X, y, epochs, lr)
{
  X_mat <- cbind(1, X)
  beta_hat <- matrix(1, nrow=ncol(X_mat))
  for (j in 1:epochs)
  {
    residual <- sigmoid(X_mat %*% beta_hat) - y
    # Update weights with gradient
    delta <- t(X_mat) %*% as.matrix(residual, ncol=nrow(X_mat)) *  (1/nrow(X_mat))
    beta_hat <- beta_hat - (lr*delta)
  }
  # Print log-likliehood
  print(log_likelihood(X_mat, y, beta_hat))
  # Return
  beta_hat
}

# Takes a while to converge!
beta_hat <- logistic_reg(X, y, 1000000, 10)
print(beta_hat)
# -42.63778
# Sepal.Length -19.47524
# Sepal.Width  -52.77898
# Petal.Length  74.49212
# Petal.Width  144.46041