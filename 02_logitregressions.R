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
y <- ifelse(data_df$Species=="virginica", 1, 0)

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


# Takes a while to converge!
beta_hat <- logistic_reg(X, y, 300000, 5)
print(beta_hat)

# Intercept    -36.46195
# Sepal.Length -28.80112
# Petal.Length  95.94318

# Visualise the decision boundary
intcp <- beta_hat[1]/-(beta_hat[3])
slope <- beta_hat[2]/-(beta_hat[3])

abline(intcp , slope, col='purple')
