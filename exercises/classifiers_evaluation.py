import os

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from math import atan2, pi

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, f"../datasets/{filename}")
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        p = Perceptron(callback=lambda p_, X_, y_: losses.append(p_.loss(X, y)))
        p.fit(X, y)

        # Plot figure
        fig = go.Figure(go.Scatter(x=list(range(len(losses))), y=losses))
        fig.update_layout(
            title=f"Perceptron loss on the '{n}' dataset",
            xaxis_title="Iteration",
            yaxis_title="Loss"
        )
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lda_pred = lda.predict(X)
        bayes = GaussianNaiveBayes()
        bayes.fit(X, y)
        bayes_pred = bayes.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Accuracy of Bayes: {accuracy(y, bayes_pred):.4f}",
                                                            f"Accuracy of LDA: {accuracy(y, lda_pred):.4f}"])
        fig.update_layout(showlegend=False, title_text=f"LDA and GaussianNaiveBayes vs. {f}", xaxis_title="Feature 1",
                          yaxis_title="Feature 2")

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=lda_pred, symbol=y)),
            row=1, col=2
        )
        for mu in lda.mu_:
            fig.add_trace(
                get_ellipse(mu, lda.cov_), row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=[mu[0]], y=[mu[1]], mode="markers", marker_symbol=34, marker_line_color="black",
                           marker_line_width=2, marker_size=15), row=1, col=2
            )
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=bayes_pred, symbol=y)),
            row=1, col=1
        )
        for mu, cov in zip(bayes.mu_, bayes.vars_):
            fig.add_trace(
                get_ellipse(mu, np.diag(cov)), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=[mu[0]], y=[mu[1]], mode="markers", marker_symbol=34, marker_line_color="black",
                           marker_line_width=2, marker_size=15), row=1, col=1
            )
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
