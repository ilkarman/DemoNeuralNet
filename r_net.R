sigmoid <- function(z){1.0/(1.0+exp(-z))}

sigmoid_prime <- function(z){sigmoid(z)*(1-sigmoid(z))}

neuralnetwork <- function(sizes)
{
  num_layers <- length(sizes)
  biases <- sapply(sizes[-1], function(f) {matrix(rnorm(n=f), nrow=f, ncol=1)})
  weights <- sapply(list(sizes[1:length(sizes)-1], sizes[-1]), function(f) {
    matrix(rnorm(n=f[1]*f[2]), nrow=f[2], ncol=f[1])})
}

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
SGD <- function(training_data, epochs, mini_batch_size, lr)
{
  #epochs <- 10 # Debug
  #mini_batch_size <- 2 # Debug
  n <- length(training_data)
  for (j in 1:epochs){
    # Stochastic mini-batch
    training_data <- sample(training_data)
    # Partition set into mini-batches
    mini_batches <- split(training_data, 
                          ceiling(seq_along(training_data)/mini_batch_size))
    # Feed forward all mini-batch
    for (k in 1:length(mini_batches)) {
      update_mini_batch(mini_batches[[k]])
    }
    # Logging
    cat("Epoch: ", j, " complete")
  }
}

update_mini_batch <- function(mini_batch, lr)
{
  mini_batch <- mini_batches[[1]]  # DEBUG
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
    
    # Create function (TODO!)
    delta_nablas <- backprop(x, y)
    
    delta_nabla_b <- delta_nablas[[1]]
    delta_nabla_w <- delta_nablas[[-1]]
    nabla_b <- lapply(seq_along(biases),function(j)
      unlist(nabla_b[j])+unlist(delta_nabla_b[j]))
    nabla_w <- lapply(seq_along(weights),function(j)
      unlist(nabla_w[j])+unlist(delta_nabla_w[j]))
  }
  # Opposite direction of gradient
  # TODO!!
  # This should overwrite global (i.e. self.weights)
  weights <- lapply(seq_along(weights), function(j)
    unlist(weights[j])-(lr/nmb)*unlist(nabla_w[j]))
  biases <- lapply(seq_along(biases), function(j)
    unlist(biases[j])-(lr/nmb)*unlist(nabla_b[j]))
}

# TODO!
backprop <- function(x, y)
{
  
}

# TODO!
evaluate <- function(test_data)

cost_derivative <- function(output_activations, y){output_activations-y}

###############################
## EVALUATE (Compare to Python)
## I feed in the below values to python and
## check that R script matches
###############################

sizes <- c(3, 5, 2)

biases <- list(
  matrix(c(-0.28080838, 
           0.2316751,
           0.87261225,
           -0.96765989,
           -1.868544), 
         nrow=5, ncol=1, byrow=TRUE),
  matrix(c(2.10775394,
           0.41855275), 
         nrow=2, ncol=1, byrow=TRUE)
  )

weights <- list(
  matrix(c(-0.81267026, 0.17627318, -0.60639905, 
           0.50974091, 2.34693197, 0.33875867,
           -1.20632438, -1.25457351, -1.17803266,
           0.06163412, 0.61925722,  0.87939343, 
           -0.41764508, -0.28984466,  0.09663896), 
         nrow=5, ncol=3, byrow = TRUE),
  matrix(c(0.37480004, 0.04123139, 1.5200263, -2.02504715, 0.2665885, 
           1.1946554, 0.18426967, -0.16337889, -0.91305046, 0.05401374), 
         nrow=2, ncol=5, byrow = TRUE))

# Should be:
#(2, 3)
#[[ 0.92956719  0.92438907  0.91830857],
#[ 0.64309165  0.6703279   0.63000043]]
result_test <- feedforward(0.5)
dim(result_test)
result_test # CORRECT!

# Example of train_data
train_data_0 <- list(
  c(0.88607595, 
     0.40506329,
     0.59493671,
     0.17721519),
  c(0,1,0))
train_data_1 <- list(
  c(0.70886076, 
    0.37974684,
    0.56962025,
    0.18987342),
  c(0,1,0))

training_data <- list(train_data_0, train_data_1)

