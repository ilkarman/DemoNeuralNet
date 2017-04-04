# logitregressions.R
# Neural Network from scratch in R
# Ilia 03.04.2017

set.seed(1234567)

#####################################################
# 1. Logistic Regression in R
#####################################################

# Two possible outcomes -> binomial
data_df <- as.data.frame(iris)
idx <- data_df$Species %in% c("virginica", "versicolor")
data_df <- data_df[idx,]
y <- ifelse(data_df$Species=="virginica",1,0)

# For faster convergence let's rescale X
# So that we can plot this consider only 2 variables
X <- data_df[c(1,3)]
X <- as.matrix(X/max(X))

# Fit model
model <- glm(y ~ X, family=binomial(link='logit'))

# Params
print(coef(model))
# Coefficients:
# (Intercept) XSepal.Length XPetal.Length 
# -39.83851     -31.73243     105.16992 
summary(model)

# Visualise the decision boundary
intcp <- coef(model)[1]/-(coef(model)[3])
slope <- coef(model)[2]/-(coef(model)[3])

# Our points
plot(x=X[,1], y=X[,2], cex = 1, col=data_df$Species,
     main = "Iris type by length and width", 
     xlab = "Sepal Length", ylab = "Petal Length")
legend(x='topright', legend=unique(data_df$Species),col=unique(data_df$Species), pch=1)
# Decision boundary
abline(intcp , slope, col='blue')

#####################################################
# 2. Logistic Regression with GD
#####################################################

# Calculate activation function (sigmoid for logit)
sigmoid <- function(z){1.0/(1.0+exp(-z))}

# Calculate log-likelihood (easier to max than likelihood)
log_likelihood <- function(X_mat, y, beta_hat)
{
  scores <- X_mat %*% beta_hat
  # Need to broadcast (y %*% scores)
  ll <- rep(y %*% scores, nrow(X_mat)) - log(1+exp(scores))
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
beta_hat <- logistic_reg(X, y, 300000, 3)
print(beta_hat)

# Intercept    -36.46195
# Sepal.Length -28.80112
# Petal.Length  95.94318

# Visualise the decision boundary
intcp <- beta_hat[1]/-(beta_hat[3])
slope <- beta_hat[2]/-(beta_hat[3])

abline(intcp , slope, col='purple')
