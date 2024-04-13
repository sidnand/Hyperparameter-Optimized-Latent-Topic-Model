data {
  int<lower=1> n;  // Vocabulary size
  int<lower=1> d;  // Number of labels
  int<lower=1> m;  // Number of topics

  matrix[n, d] X;  // Word-label matrix
  real<lower=0> alpha;  // Tuning parameter 
}

parameters {
  simplex[m] topic_dist;  // Topic distribution (uniform prior)
  real<lower=0>[n, m] theta;  // Word-topic intensity (continuous)
}

model {
  topic_dist ~ uniform([m]); // Uniform prior over topics 
  for (i in 1:n) {
    for (k in 1:m) {
      target += alpha * theta[i, k] * sum(X[i]);  // Log-likelihood contribution
    }
  }
}
