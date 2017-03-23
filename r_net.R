sigmoid <- function(z){1.0/(1.0+exp(-z))}

sigmoid_prime <- function(z){sigmoid(z)*(1-sigmoid(z))}

neuralnetwork <- function(sizes)
{
  sizes <- c(3, 5, 2)
  num_layers <- length(sizes)
  biases <- sapply(sizes[-1], function(f) {matrix(rnorm(n=f), nrow=f, ncol=1)})
  weights <- sapply(list(sizes[1:length(sizes)-1], sizes[-1]), function(f) {
    matrix(rnorm(n=f[1]*f[2]), nrow=f[2], ncol=f[1])
  })
}

feedforward <- function(a)
{
  for (f in 1:length(biases)){
    b <- biases[[f]]
    w <- weights[[f]]
    # Equivalent of python np.dot(w,a)
    w_a <- if(is.null(dim(a))) w*a else w%*%a
    # Need to manually broadcast b to conform to np.dot(w,a)
    b_broadcast <- matrix(b, nrow=dim(w_a)[1], ncol=dim(w_a)[-1])
    a <- sigmoid(w_a + b_broadcast)
  }
  a
}

# Complete (and tested up to this point) ...

SGD <- function(something)
{
  # TODO
}

###############################
## EVALUATE (Compare to Python)
###############################

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
result_test

# CORRECT!
