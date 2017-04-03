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
data_df$y <- ifelse(data_df$Species=="virginica",1,0)
data_df$Species <- NULL

# Fit model
model <- glm(y ~., family=binomial(link='logit'), data=data_df)

# Params
print(model)
# Coefficients:
# (Intercept)  Sepal.Length   Sepal.Width  Petal.Length   Petal.Width  
# -42.638        -2.465        -6.681         9.429        18.286 
summary(model)

#####################################################
# 2. Logistic Regression with GD
#####################################################

X <- data_df[c(1:4)]
y <- data_df[-c(1:4)]

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
  X_mat <- cbind(1, as.matrix(X))
  beta_hat <- matrix(1, nrow=ncol(X_mat))
  for (j in 1:epochs)
  {
    residual <- sigmoid(X_mat %*% beta_hat) - y
    # Update weights with gradient
    delta <- t(X_mat) %*% as.matrix(residual, ncol=nrow(X_mat))
    beta_hat <- beta_hat - (lr*delta)
  }
  # Print log-likliehood
  #print(log_likelihood(X, y, beta_hat))
  # Return
  beta_hat
}

beta_hat <- logistic_reg(X, y, 300000, 0.3)
