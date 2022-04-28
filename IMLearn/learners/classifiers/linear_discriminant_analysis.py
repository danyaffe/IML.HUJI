from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        k = self.classes_.size
        self.pi_ = np.zeros((k,))
        self.mu_ = np.zeros((k, X.shape[1]))
        for i, c in enumerate(self.classes_):
            self.pi_[i] = np.sum(c == y) / k
            self.mu_[i, :] = (X[y == c]).mean(axis=0)
        mu_y = self.mu_[y]
        self.cov_ = np.sum(np.dstack([np.outer(x_i - mu_y_i, x_i - mu_y_i) for x_i, mu_y_i in zip(X, mu_y)]),
                           axis=2) / (X.shape[0] - k)
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        res = np.zeros((X.shape[0],))
        for i, x in enumerate(X):
            res[i] = self.classes_[
                np.argmax([(self._cov_inv @ self.mu_[k]).transpose() @ x + np.log(self.pi_[k]) - .5 *
                           self.mu_[k] @ self._cov_inv @ self.mu_[k] for k, class_ in enumerate(self.classes_)])
            ]
        return res

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        d = X.shape[0]
        log = np.log
        ret = np.zeros((d, self.classes_.size))
        for c in self.classes_:
            temp = log(self.pi_[c]) - d / 2 * log(2 * np.pi) - .5 * log(det(self.cov_))
            for i, x in enumerate(X):
                ret[i, c] = temp - .5 * (x - self.mu_[c]).transpose() @ self._cov_inv @ (x - self.mu_[c])

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
