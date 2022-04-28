import os

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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
        fig.show()


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
        bayes.fit(X,y)
        bayes_pred = bayes.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2)
        fig.update_layout(showlegend=False,title_text="Side By Side Subplots")

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=lda_pred, symbol=y)),
            row=1, col=1
        )


        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=bayes_pred, symbol=y)),
            row=1, col=2
        )

        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
