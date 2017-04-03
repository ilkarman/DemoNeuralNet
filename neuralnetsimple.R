# neuralnetsimple.R
# Neural Network from scratch in R
# Ilia 03.04.2017

##########
# Issues:
# 1. Running Breast-Cancer after iris doesn't seem to work well (as if something remains in memory from iris)
##########

##########
# To-do:
# 1. Add Cross-Entropy loss
# 2. Add regularization
# 3. Add ReLU and Tanh activations
# 4. Allow saving/loading trained models
# 5. Include more comments
# 6. Include example on MNIST
# 7. Allow better logging
##########

################################################
## DATA FUNCTIONS
################################################

# Return train, test lists in format for NN from input-dataframe
train_test_from_df <- function(df, predict_col_index, train_ratio)
{
  # Helper functions
  # Function to encode factor column as N-dummies
  dmy <- function(df)
  {
    # Select only factor columns
    factor_columns <- which(sapply(df, is.factor))
    if (length(factor_columns) > 0)
    {
      # Split factors into dummies
      dmy_enc <- model.matrix(~. + 0, data=df[factor_columns], 
                              contrasts.arg = lapply(df[factor_columns], contrasts, contrasts=FALSE))
      dmy_enc <- as.data.frame(dmy_enc)
      # Attach factors to df
      df <- cbind(df, dmy_enc)
      # Delete original columns
      df[c(factor_columns)] <- NULL
    }
    df
  }
  
  # Function to standarise inputs to range(0, 1)
  scalemax <- function(df)
  {
    numeric_columns <- which(sapply(df, is.numeric))
    if (length(numeric_columns)){df[numeric_columns] <- lapply(df[numeric_columns], function(x){x/max(x)})}
    df
  }
  
  # Function to convert df to list of rows
  listfromdf <- function(df){as.list(as.data.frame(t(df)))}
  
  
  # Omit NAs (allow other options later)
  df <- na.omit(df)
  # Get list for X-data
  X_data <- listfromdf(dmy(scalemax(df[-c(predict_col_index)])))
  # Get list for y-data
  y_data <- listfromdf(dmy(df[c(predict_col_index)]))
  # Combine X,y
  all_data <- list()
  for (i in 1:length(X_data)){
    all_data[[i]] <- c(X_data[i], y_data[i])
  }
  # Shuffle before splitting
  all_data <- sample(all_data)
  # Split to training and test
  tr_n <- round(length(all_data)*train_ratio)
  # Return (training, testing)
  list(all_data[c(1:tr_n)], all_data[-c(1:tr_n)])
}

################################################
## NN FUNCTIONS
################################################

# Calculate activation function
sigmoid <- function(z){1.0/(1.0+exp(-z))}

# Partial derivative of activation function
sigmoid_prime <- function(z){sigmoid(z)*(1-sigmoid(z))}

# Cost function derivative
cost_derivative <- function(output_activations, y){output_activations-y}

# Init neural-network
neuralnetwork <- function(sizes)
{
  num_layers <- length(sizes)
  listw <- sizes[1:length(sizes)-1] # Skip last (weights from 1st to 2nd-to-last)
  listb <-  sizes[-1]  # Skip first element (biases from 2nd to last)
  
  # Initialise with gaussian distribution for biases and weights
  biases <- lapply(seq_along(listb), function(idx){
    r <- listb[[idx]]
    matrix(rnorm(n=r), nrow=r, ncol=1)
  })
  weights <- lapply(seq_along(listb), function(idx){
    c <- listw[[idx]]
    r <- listb[[idx]]
    matrix(rnorm(n=r*c), nrow=r, ncol=c)
  })
  
  # Return
  list(sizes, num_layers, biases, weights)
}

# Feedforward input to get prediction (for socring)
feedforward <- function(a)
{
  for (f in 1:length(biases)){
    a <- matrix(a, nrow=length(a), ncol=1)
    b <- biases[[f]]
    w <- weights[[f]]
    # (py) a = sigmoid(np.dot(w, a) + b)
    # Equivalent of python np.dot(w,a)
    w_a <- w%*%a
    # Need to manually broadcast b to conform to np.dot(w,a)
    b_broadcast <- matrix(b, nrow=dim(w_a)[1], ncol=dim(w_a)[-1])
    a <- sigmoid(w_a + b_broadcast)
  }
  a
}

# Train network using stochastic-gradient descent
SGD <- function(training_data, epochs, mini_batch_size, lr, biases, weights)
{
  # Every epoch
  for (j in 1:epochs){
    # Stochastic mini-batch (shuffle data)
    training_data <- sample(training_data)
    # Partition set into mini-batches
    mini_batches <- split(training_data, 
                          ceiling(seq_along(training_data)/mini_batch_size))
    # Feed forward (and back) all mini-batches
    for (k in 1:length(mini_batches)) {
      # Update biases and weights
      res <- update_mini_batch(mini_batches[[k]], lr, biases, weights)
      # Logging
      #cat("Updated: ", k, " mini-batches")
      biases <- res[[1]]
      weights <- res[[-1]]
    }
    # Logging
    #cat("Epoch: ", j, " complete")
  }
  cat("Training complete")
  # Return trained biases and weights
  list(biases, weights)
}

update_mini_batch <- function(mini_batch, lr, biases, weights)
{
  nmb <- length(mini_batch)
  listw <- sizes[1:length(sizes)-1] 
  listb <-  sizes[-1]  
  
  # Initialise updates with zero vectors (for EACH mini-batch)
  nabla_b <- lapply(seq_along(listb), function(idx){
    r <- listb[[idx]]
    matrix(0, nrow=r, ncol=1)
  })
  nabla_w <- lapply(seq_along(listb), function(idx){
    c <- listw[[idx]]
    r <- listb[[idx]]
    matrix(0, nrow=r, ncol=c)
  })  
  
  # Go through mini_batch
  for (i in 1:nmb){
    x <- mini_batch[[i]][[1]]
    y <- mini_batch[[i]][[-1]]
    # Back propogation will return delta
    # Backprop for each obeservation in mini-batch
    delta_nablas <- backprop(x, y, biases, weights)
    delta_nabla_b <- delta_nablas[[1]]
    delta_nabla_w <- delta_nablas[[-1]]
    # Add on deltas to nabla
    # The python equivalent for below is:
    # nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
    # nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    nabla_b <- lapply(seq_along(biases),function(j)
      unlist(nabla_b[[j]])+unlist(delta_nabla_b[[j]]))
    nabla_w <- lapply(seq_along(weights),function(j)
      unlist(nabla_w[[j]])+unlist(delta_nabla_w[[j]]))
  }
  # After mini-batch has finished update biases and weights:
  # i.e. weights = weights - (learning-rate/numbr in batch)*nabla_weights
  # Opposite direction of gradient
  weights <- lapply(seq_along(weights), function(j)
    unlist(weights[[j]])-(lr/nmb)*unlist(nabla_w[[j]]))
  biases <- lapply(seq_along(biases), function(j)
    unlist(biases[[j]])-(lr/nmb)*unlist(nabla_b[[j]]))
  # Return
  list(biases, weights)
}

# Return gradient updates using back-propogation
backprop <- function(x, y, biases, weights)
{
  # Initialise updates with zero vectors
  listw <- sizes[1:length(sizes)-1] 
  listb <-  sizes[-1]  
  
  # Initialise updates with zero vectors (for EACH mini-batch)
  nabla_b_backprop <- lapply(seq_along(listb), function(idx){
    r <- listb[[idx]]
    matrix(0, nrow=r, ncol=1)
  })
  nabla_w_backprop <- lapply(seq_along(listb), function(idx){
    c <- listw[[idx]]
    r <- listb[[idx]]
    matrix(0, nrow=r, ncol=c)
  })  
  
  # First:
  # Feed-forward (get predictions)
  activation <- matrix(x, nrow=length(x), ncol=1)
  activations <- list(matrix(x, nrow=length(x), ncol=1))
  # z = f(w.x + b)
  # So need zs to store all z-vectors
  zs <- list()
  for (f in 1:length(biases)){
    b <- biases[[f]]
    w <- weights[[f]]
    w_a <- w%*%activation
    b_broadcast <- matrix(b, nrow=dim(w_a)[1], ncol=dim(w_a)[-1])
    z <- w_a + b
    zs[[f]] <- z
    activation <- sigmoid(z)
    activations[[f+1]] <- activation  # Activations already contain one element
  }
  # Second:
  # Backwards (update gradient using errors)
  # Last layer
  sp <- sigmoid_prime(zs[[length(zs)]])
  # Gradients update depending on how big the error in prediction was
  delta <- cost_derivative(activations[[length(activations)]], y) * sp
  nabla_b_backprop[[length(nabla_b_backprop)]] <- delta
  nabla_w_backprop[[length(nabla_w_backprop)]] <- delta %*% t(activations[[length(activations)-1]])
  # Second to second-to-last-layer
  for (k in 2:(num_layers-1)) {
    sp <- sigmoid_prime(zs[[length(zs)-(k-1)]])
    delta <- (t(weights[[length(weights)-(k-2)]]) %*% delta) * sp
    nabla_b_backprop[[length(nabla_b_backprop)-(k-1)]] <- delta
    testyy <- t(activations[[length(activations)-k]])
    nabla_w_backprop[[length(nabla_w_backprop)-(k-1)]] <- delta %*% testyy
  }
  # return (nabla_b, nabla_w)
  return_nabla <- list(nabla_b_backprop, nabla_w_backprop)
  return_nabla
}

# Evaluate test-data
evaluate <- function(testing_data, biases, weights)
{
  predictions <- list()
  truths <- list()
  # Probably can avoid the for-loop here but this function run only once
  for (i in 1:length(testing_data)){
    test_data_chunk <- testing_data[[i]]
    test_x <- test_data_chunk[[1]]
    test_y <- test_data_chunk[[-1]]
    predictions[i] <- which.max(feedforward(test_x))
    truths[i] <- which.max(test_y)
  }
  correct <- sum(mapply(function(x,y) x==y, predictions, truths))
  total <- length(testing_data)
  # Print accuracy
  print(correct/total)
  # Return confusion
  res <- as.data.frame(cbind(t(as.data.frame(predictions)), t(as.data.frame(truths))))
  colnames(res) <- c("Prediction", "Truth")
  table(as.vector(res$Prediction), as.vector(res$Truth))
}


################################################
## RUN (Example Iris)
################################################
# Example on iris data
train_test_split <- train_test_from_df(df = iris, predict_col_index = 5, train_ratio = 0.5)
training_data <- train_test_split[[1]]
testing_data <- train_test_split[[2]]

input_neurons <- length(training_data[[1]][[1]])
output_neurons <- length(training_data[[1]][[-1]])

# Step 1. Initialise nueral network (bias and weights for layers)
# Tested only with one hidden layer
create_neural_net <- neuralnetwork(c(input_neurons, 6, 4, output_neurons))

sizes <- create_neural_net[[1]]
num_layers <- create_neural_net[[2]]
biases <- create_neural_net[[3]]
weights <- create_neural_net[[4]]

# Step 2. Train NN using SGD
trained_net <- SGD(training_data=training_data,
                   epochs=1000, 
                   mini_batch_size=10,
                   lr=0.2,
                   biases=biases, 
                   weights=weights)

# Trained matricies:
biases <- trained_net[[1]]
weights <- trained_net[[-1]]

# Accuracy (train)
evaluate(training_data, biases, weights)
# Accuracy (test)
evaluate(testing_data, biases, weights)

################################################
## RUN (Example Breast Cancer)
################################################
#install.packages('mlbench')
library(mlbench)
data("BreastCancer")
df_in <- as.data.frame(BreastCancer)
df_in$Id <- NULL

train_test_split <- train_test_from_df(df = df_in, predict_col_index = 10, train_ratio = 0.5)
training_data <- train_test_split[[1]]
testing_data <- train_test_split[[2]]

input_neurons <- length(training_data[[1]][[1]])
output_neurons <- length(training_data[[1]][[-1]])

# Step 1. Initialise nueral network (bias and weights for layers)
# Tested only with one hidden layer
create_neural_net <- neuralnetwork(c(input_neurons, 6, 4, output_neurons))

sizes <- create_neural_net[[1]]
num_layers <- create_neural_net[[2]]
biases <- create_neural_net[[3]]
weights <- create_neural_net[[4]]

# Step 2. Train NN using SGD
trained_net <- SGD(training_data=training_data,
                   epochs=100, 
                   mini_batch_size=50,
                   lr=0.2,
                   biases=biases, 
                   weights=weights)

# Trained matricies:
biases <- trained_net[[1]]
weights <- trained_net[[-1]]

# Accuracy (train)
evaluate(training_data, biases, weights)
# Accuracy (test)
evaluate(testing_data, biases, weights)



