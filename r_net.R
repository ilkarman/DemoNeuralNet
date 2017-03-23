# Seems to actually run ... happy surprise

# NOTE to self: 
#1. [-1] in R removes first element NOT takes last
#2. for (i in 1:2) in R does include 2

################################################
## Load Data
################################################
library(caret)

data_input <- as.data.frame(iris)

# X (scale)
scalemax <- function(x){x/max(x)}
x_data <- as.list(as.data.frame(t(scalemax(data_input[c(1:4)]))))
# y (one-hot-encode)
dmy <- dummyVars(" ~ Species", data=data_input)
y_data <- as.list(as.data.frame(t(predict(dmy, newdata = data_input))))

# Create full data vector
all_data <- list()
for (i in 1:length(x_data)){
  all_data[[i]] <- c(x_data[i], y_data[i])
}

# Shufle
all_data <- sample(all_data)
# Split to training and test
training_data <- all_data[1:100]
testing_data <- all_data[101:150]

################################################
## RUN
################################################
create_neural_net <- neuralnetwork(c(4, 6, 3))

sizes <- create_neural_net[[1]]
num_layers <- create_neural_net[[2]]
biases <- create_neural_net[[3]]
weights <- create_neural_net[[4]]

trained_net <- SGD(training_data=training_data,
                   epochs=1000, 
                   mini_batch_size=10,
                   lr=0.3,
                   biases=biases, 
                   weights=weights)

print("Biase After: ")
biases <- trained_net[[1]]
#print(biases)
print("Weights After: ")
weights <- trained_net[[-1]]
#print(weights)

# Accuracy
evaluate(testing_data, biases, weights)

################################################
## FUNCTIONS
################################################

sigmoid <- function(z){1.0/(1.0+exp(-z))}

sigmoid_prime <- function(z){sigmoid(z)*(1-sigmoid(z))}

cost_derivative <- function(output_activations, y){output_activations-y}

# Init neural-network
neuralnetwork <- function(sizes)
{
  num_layers <- length(sizes)
  biases <- sapply(sizes[-1], function(f) {matrix(rnorm(n=f), nrow=f, ncol=1)})
  weights <- sapply(list(sizes[1:length(sizes)-1], sizes[-1]), function(f) {
    matrix(rnorm(n=f[1]*f[2]), nrow=f[2], ncol=f[1])})
  # DEBUG
  # Initialise at 0.5 to compare to python script
  #biases <- sapply(sizes[-1], function(f) {matrix(0.5, nrow=f, ncol=1)})
  #weights <- sapply(list(sizes[1:length(sizes)-1], sizes[-1]), function(f) {
  #  matrix(0.5, nrow=f[2], ncol=f[1])})
  # DEBUG
  # Return
  list(sizes, num_layers, biases, weights)
}

# FF for scoring
feedforward <- function(a)
{
  for (f in 1:length(biases)){
    a <- matrix(a, nrow=length(a), ncol=1)
    b <- biases[[f]]
    w <- weights[[f]]
    # Seriously? All I want to replicate is:
    # (py) a = sigmoid(np.dot(w, a) + b)
    
    # Equivalent of python np.dot(w,a)
    w_a <- if(is.null(dim(a))) w*a else w%*%a
    # Need to manually broadcast b to conform to np.dot(w,a)
    b_broadcast <- matrix(b, nrow=dim(w_a)[1], ncol=dim(w_a)[-1])
    a <- sigmoid(w_a + b_broadcast)
  }
  a
}

# Complete (and tested up to this point) ...

# Ignore validation data for now
SGD <- function(training_data, epochs, mini_batch_size, lr, biases, weights)
{
  n <- length(training_data)
  for (j in 1:epochs){
    # Stochastic mini-batch
    training_data <- sample(training_data)
    # Partition set into mini-batches
    mini_batches <- split(training_data, 
                          ceiling(seq_along(training_data)/mini_batch_size))
    # Feed forward all mini-batch
    for (k in 1:length(mini_batches)) {
      res <- update_mini_batch(mini_batches[[k]], lr, biases, weights)
      # Logging
      #cat("Updated: ", k, " mini-batches")
      biases <- res[[1]]
      weights <- res[[-1]]
    }
    # Logging
    cat("Epoch: ", j, " complete")
  }
  # Return
  list(biases, weights)
}

update_mini_batch <- function(mini_batch, lr, biases, weights)
{
  nmb <- length(mini_batch)
  # Initialise updates with zero vectors
  nabla_b <- sapply(sizes[-1], function(f) {matrix(0, nrow=f, ncol=1)})
  nabla_w <- sapply(list(sizes[1:length(sizes)-1], sizes[-1]), function(f) {
    matrix(0, nrow=f[2], ncol=f[1])})
  # Go through mini_batch
  for (i in 1:nmb){
    x <- mini_batch[[i]][[1]]
    y <- mini_batch[[i]][[-1]]
    # Back propogatoin will return delta
    delta_nablas <- backprop(x, y, biases, weights)
    delta_nabla_b <- delta_nablas[[1]]
    delta_nabla_w <- delta_nablas[[-1]]
    
    # The python is:
    # nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
    # nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    nabla_b <- lapply(seq_along(biases),function(j)
      unlist(nabla_b[[j]])+unlist(delta_nabla_b[[j]]))
    nabla_w <- lapply(seq_along(weights),function(j)
      unlist(nabla_w[[j]])+unlist(delta_nabla_w[[j]]))
  }
  
  # Opposite direction of gradient
  # TODO!!
  # This should overwrite global (i.e. self.weights)
  # I understand below is wrong, but need help to rewrite as OO
  # Hmm I think these are wrong?
  # Fix thesE:
  weights <- lapply(seq_along(weights), function(j)
    unlist(weights[[j]])-(lr/nmb)*unlist(nabla_w[[j]]))
  biases <- lapply(seq_along(biases), function(j)
    unlist(biases[[j]])-(lr/nmb)*unlist(nabla_b[[j]]))
  # Return
  list(biases, weights)
}

backprop <- function(x, y, biases, weights)
{

  # Initialise updates with zero vectors
  nabla_b_backprop <- sapply(sizes[-1], function(f) {matrix(0, nrow=f, ncol=1)})
  nabla_w_backprop <- sapply(list(sizes[1:length(sizes)-1], sizes[-1]), function(f) {
    matrix(0, nrow=f[2], ncol=f[1])})
  
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
    # Activations already contains init element
    activations[[f+1]] <- activation
  }
  
  # Backwards (update gradient using errors)
  # Last layer
  sp <- sigmoid_prime(zs[[length(zs)]])
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

evaluate <- function(testing_data, biases, weights)
{
  predictions <- list()
  truths <- list()
  for (i in 1:length(testing_data)){
    test_data_chunk <- testing_data[[i]]
    test_x <- test_data_chunk[[1]]
    test_y <- test_data_chunk[[-1]]
    predictions[i] <- which.max(feedforward(test_x))
    truths[i] <- which.max(test_y)
  }
  correct <-sum(mapply(function(x,y) x==y, predictions, truths))
  total <- length(testing_data)
  # Return accuracy
  correct/total
}


