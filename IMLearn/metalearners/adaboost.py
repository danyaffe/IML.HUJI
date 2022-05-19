import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from ..learners.classifiers import DecisionStump as Stump
from ..metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.D_ = np.ones_like(y) / y.size
        self.models_ = []
        self.weights_ = []
        for i in range(self.iterations_):
            s = Stump()
            s.fit(X, self.D_ * y)
            total_error = s.loss(X, self.D_ * y)
            self.weights_.append(self._amount_of_say(total_error))
            self._update_distribution(s.predict(X) == y, self.weights_[i])
            self.models_.append(s)
        self.weights_ = np.array(self.weights_)

    def _amount_of_say(self, total_error):
        total_error = min(max(total_error, 0.000001), 0.999999)
        return 1 / 2 * np.log((1 - total_error) / total_error)

    def _update_distribution(self, correctly_classified, amount_of_say):
        self.D_[~correctly_classified] *= np.exp(amount_of_say)
        self.D_[correctly_classified] *= np.exp(-amount_of_say)
        self.D_ /= self.D_.sum()

    def _predict(self, X):
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
        return self.partial_predict(X, self.iterations_)

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
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # models_weights = np.array((np.array(self.models_[:T]), self.weights_[:T])).T
        # map_ = map(lambda m_w: m_w[0].predict(X) * m_w[1], models_weights)
        T = max(T,1)
        sum_ = np.sum(np.array([m.predict(X) * w for m, w in zip(self.models_[:T], self.weights_[:T])]), axis=0)
        return np.sign(sum_)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.partial_predict(X, T))
