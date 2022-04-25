from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    samples = np.random.normal(mu, var, 1000)
    univariate_model = UnivariateGaussian()
    univariate_model.fit(samples)
    print(f"({univariate_model.mu_}, {univariate_model.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    expectation_distances = []
    x = range(10, 1010, 10)
    for batch in x:
        univariate_model.fit(samples[:batch])
        expectation_distances.append(np.abs(univariate_model.mu_ - mu))
    plt.plot(x, expectation_distances)
    plt.title("Absolute distance between real and estimated $\mu$ values")
    plt.xlabel("sample size")
    plt.ylabel("distance")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = univariate_model.pdf(samples)
    plt.scatter(samples, pdf, s=.1)
    plt.title("PDF of sampled values")
    plt.xlabel("probability")
    plt.ylabel("value")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = (np.array([0, 0, 4, 0])).transpose()
    cov = np.array([[1, .2, 0, .5], [.2, 2, 0, 0], [0, 0, 1, 0], [.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, cov, 1000)
    multivariate_model = MultivariateGaussian()
    multivariate_model.fit(samples)
    print(multivariate_model.mu_)
    print(multivariate_model.cov_)

    # Question 5 - Likelihood evaluation
    f = np.linspace(-10, 10, 200)
    table = []
    for f_1 in tqdm(f):
        row = []
        for f_3 in f:
            mu_2 = np.array([f_1, 0, f_3, 0]).transpose()
            row.append(MultivariateGaussian.log_likelihood(mu_2, cov, samples))
        table.append(row)

    table = np.asarray(table)
    fig, ax = plt.subplots()
    z_min, z_max = table.min(), table.max()
    c = ax.pcolormesh(f, f, table, cmap='inferno', vmin=z_min, vmax=z_max)
    ax.axis([f.min(), f.max(), f.min(), f.max()])
    fig.colorbar(c, ax=ax)
    plt.xlabel("$f_3$")
    plt.ylabel("$f_1$")
    plt.title("Heatmap of log-likelihood values with $\mu=[f_1,0,f_3,0]^T$")
    plt.show()

    # Question 6 - Maximum likelihood
    argmax = table.argmax()
    print(f"The best model was (f_1,f_3): ({f[argmax//len(f)]:0.3f},{f[argmax % len(f)]:0.3f})")

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
