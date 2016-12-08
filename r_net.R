# Attempted rewrite of py_net.py

sigmoid <- function(z){1.0/(1.0+exp(-z))}
sigmoid_prime <- function(z){sigmoid(z)*(1-sigmoid(z))}

neuralnetwork <- function(sizes)
  {
    num_layers = length(sizes)
    biases <- sapply(sizes[-1], function(f) {matrix(rnorm(n=f*1))})
    weights <- sapply(list(sizes[1:length(sizes)-1], sizes[-1]), function(f) {
      matrix(rnorm(n=f[1]*f[2]))
      })
    # How to combine these??
    list(num_layers, sizes, biases, weights)
  }






# Test
sample_size = c(3, 5, 2)
sample_net <- neuralnetwork(sample_size)