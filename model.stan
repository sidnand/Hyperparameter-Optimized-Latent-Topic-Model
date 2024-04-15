data {
  int<lower=1> n;  // Number of words
  int<lower=1> d;  // Number of documents
  int<lower=0> X[d, n]; // Word-document matrix (assuming word counts)

  int<lower=1> m;  // Number of topics

  // Hyperparameters
  vector<lower=0>[m] a; // Dirichlet concentration parameter (provide some initial values)
  real<lower=0> alpha;  // Word frequency influence
  real<lower=0> beta;   // Smoothing parameter
}

parameters {
  simplex[m] theta[n];  // Topic distributions for each word
}

model {
  // Prior for topic distributions
  for (i in 1:n) {
    theta[i] ~ dirichlet(a);
  }

  // Likelihood
  for (i in 1:n) {
    for (j in 1:d) {
      for (k in 1:m) {

        // $$\log \mathbb{P}(X | \theta) = \sum_{i=1}^n \left (\alpha⋅\sum_{j=1}^{d} X_{i,j} \right) \log \left (\frac{\theta_{i,k} + \beta}{\sum_{k'=1}^{m} \theta_{i,k'} + \beta⋅m} \right)$$

        target += (alpha * X[j, i]) * log((theta[i, k] + beta) / (sum(theta[i]) + beta * m));

      }
    }
  }
}
