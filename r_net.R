# Seems to actually run ... happy surprise

# NOTE to self: 
#1. [-1] in R removes first element NOT takes last
#2. for (i in 1:2) in R does include 2

################################################
## Load Data
################################################
library(caret)

x_data <- as.list(as.data.frame(t(iris[c(1:4)])))
dmy <- dummyVars(" ~ Species", data=iris)
y_data <- as.list(as.data.frame(t(predict(dmy, newdata = iris))))

all_data <- list()
for (i in 1:length(x_data)){
  all_data[[i]] <- c(x_data[i], y_data[i])
}

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
                   lr=30,
                   biases=biases, 
                   weights=weights)

print("Biase After: ")
print(trained_net[[1]])
print("Weights After: ")
print(trained_net[[-1]])

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
  # FIX to reproduce python script
  sizes <- c(4, 6, 3)
  biases <- sapply(sizes[-1], function(f) {matrix(rnorm(n=f), nrow=f, ncol=1)})
  # FIX to reproduce python script
  biases <- list(
    matrix(c(2.21552528,0.56221113,1.29356006,0.20183309,-0.24909272,0.1880131),nrow=6, ncol=1, byrow=TRUE),
    matrix(c(-0.55172029,0.62177897,0.54538667),nrow=3, ncol=1, byrow=TRUE))
  weights <- sapply(list(sizes[1:length(sizes)-1], sizes[-1]), function(f) {
    matrix(rnorm(n=f[1]*f[2]), nrow=f[2], ncol=f[1])})
  # FIX to reproduce python script
  weights <- list(
    matrix(c(
      -1.59478751, -1.49475559, -0.28792213, -1.49318205,
      0.46775273,  0.55212408, -0.09821697,  2.16442781,
      -1.09225776, -0.50294527,  0.6827888 ,  1.26445372,
      0.05535019,  2.24894183,  1.76910817, -0.58362259,
      -0.88872193,  1.77267107,  1.91067743,  1.1684327,
      0.74813106, -0.28986976,  2.1845845 , -0.870942), nrow=6, ncol=4, byrow=TRUE),
    matrix(c(
      0.71380675, -1.1651045 , -0.44106224,
      -0.46050179, -0.6196771, 1.02890824,
      -0.04386169, -0.57472911,  0.64076485,
      -0.34151746,  0.08611568, 0.6652118,
      -1.01254895,  0.51042568, -1.09556564,
      -0.67983253,  1.23153594, -1.24757605), nrow=3, ncol=6, byrow=TRUE))
  # Return
  list(sizes, num_layers, biases, weights)
}

# FF for scoring
feedforward <- function(a)
{
  for (f in 1:length(biases)){
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
    #training_data <- sample(training_data)
    # Partition set into mini-batches
    mini_batches <- split(training_data, 
                          ceiling(seq_along(training_data)/mini_batch_size))
    # Feed forward all mini-batch
    for (k in 1:length(mini_batches)) {
      res <- update_mini_batch(mini_batches[[k]], lr, biases, weights)
      # Logging
      cat("Updated: ", k, " mini-batches")
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

# TODO!
evaluate <- function(test_data) 
{
  
}


