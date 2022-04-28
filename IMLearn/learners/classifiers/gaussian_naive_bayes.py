from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.vars_ = np.zeros((k, X.shape[1]))
        for i, c in enumerate(self.classes_):
            self.pi_[i] = np.sum(c == y) / k
            self.mu_[i, :] = (X[y == c]).mean(axis=0)
            n_k = X[y == c].shape[0]
            self.vars_[i, :] = np.sum(np.power(X - np.tile(self.mu_[i], (X.shape[0], 1)), 2)[y == c], axis=0) / (
                    n_k - 1)

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
        # res = np.zeros((X.shape[0],))
        # for i, x in enumerate(X):
        #     res[i] = self.classes_[np.argmax([-.5 * np.sum((x - self.mu_[k]).power(2) /
        #                                                    self.vars_[k]) for k, class_ in enumerate(self.classes_)])]
        # return res
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

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
            temp = log(self.pi_[c]) - d / 2 * log(2 * np.pi) - .5 * np.sum(log(self.vars_[c]))
            for i, x in enumerate(X):
                ret[i, c] = temp - np.sum(.5 * np.power((x - self.mu_[c]), 2) / self.vars_[c])
        return ret

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
