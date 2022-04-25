from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from tqdm import tqdm
import os
from typing import NoReturn, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# import plotly.graph_objects as go
# import plotly.express as px
# import plotly.io as pio
#
# pio.templates.default = "simple_white"


def load_data(filename: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.drop(["id", "date", "lat", "long"], axis=1)
    lastBuiltCol = pd.DataFrame(np.maximum(df["yr_built"].to_numpy(), df["yr_renovated"].to_numpy()),
                                columns=["yr_touched"])
    df = df.drop(["yr_built", "yr_renovated"], axis=1).join(lastBuiltCol)
    df = pd.get_dummies(df, columns=["zipcode"])
    df = df.dropna()
    df = df[(df >= 0).all(1)]
    df = df.drop(df.index[df['bedrooms'] > 20])
    y = df["price"]
    X = df.drop(["price"], axis=1)
    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    std_y = np.std(y)
    for name, values in tqdm(X.iteritems()):
        pc = float(np.cov(values, y.T)[0][1] / (np.std(values) * std_y))
        plt.scatter(values, y)
        plt.title(f"Feature correlation of {name} with $\\rho={pc:.3f}$")
        plt.savefig(output_path + "/feat_correlation_" + str(name) + ".png", format="png")
        plt.figure()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, f'../datasets/house_prices.csv')
    X, y = load_data(filename)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, dirname + "/feat_col_graphs")

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y, .75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    model = LinearRegression()
    means = []
    stds2 = []
    range_it = list(range(10, 101))
    for p in tqdm(range_it):
        losses = np.zeros((10,))
        for i in range(10):
            seed = np.random.randint(np.iinfo(np.int32).max)
            txp = train_x.sample(frac=p / 100, random_state=seed)
            typ = train_y.sample(frac=p / 100, random_state=seed)
            model.fit(txp.to_numpy(), typ.to_numpy())
            losses[i] = model.loss(test_x.to_numpy(), test_y.to_numpy())
        means.append(np.mean(losses))
        stds2.append(2 * np.std(losses))
    plt.plot(range_it, means)
    # plt.fill_between(range_it, error_min, error_max)
    plt.errorbar(range_it, means, yerr=stds2)
    plt.title("Loss of model with different train sizes")
    plt.xlabel("percentage of train used")
    plt.ylabel("Loss mean, with error")
    plt.show()
