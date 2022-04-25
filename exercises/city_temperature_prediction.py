import os

from tqdm import tqdm

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# import plotly.express as px
# import plotly.io as pio
# pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=[2])
    df = df.dropna()
    df = df[(df['Temp'] > -20)]
    dayOfYear = pd.to_datetime(df["Date"]).apply(lambda x: x.timetuple().tm_yday)
    dayOfYear = pd.DataFrame(dayOfYear).rename(columns={"Date": "DayOfYear"})
    df = df.join(dayOfYear)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, f'../datasets/City_Temperature.csv')
    Xy = load_data(filename)

    # Question 2 - Exploring data for specific country
    Isr = Xy[Xy["Country"] == "Israel"]
    years = set(Isr["Year"])
    for year in years:
        Isr_yrly = Isr[Isr["Year"] == year]
        plt.scatter(Isr_yrly["DayOfYear"], Isr_yrly["Temp"])
    plt.title("Temp by day of year, wih year color-coding")
    plt.xlabel("Day of year")
    plt.ylabel("Celsius temperature")
    plt.show()
    Isr_monthly = Isr.groupby("Month").Temp.agg(["std"])
    plt.bar(list(range(1, 13)), list(Isr_monthly["std"]))
    plt.title("std of temperature by months")
    plt.xlabel("months")
    plt.ylabel("standard deviation in Celsius")
    plt.show()

    # Question 3 - Exploring differences between countries
    country_month_df = Xy.groupby(['Country', 'Month']).Temp.agg(['mean', 'std'])
    for country in set(country_month_df.index.droplevel([1])):
        country_df = country_month_df.loc[country]
        mean = country_df["mean"]
        std = country_df["std"]
        plt.errorbar(range(1, 13), mean, yerr=std, label=country)
    plt.legend()
    plt.title("Mean and std of temperature by month for each country")
    plt.xlabel("Month")
    plt.ylabel("Temp in Celsius")
    plt.show()

    # Question 4 - Fitting model for different values of `k`
    Isr = Xy[Xy["Country"] == "Israel"]
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(Isr["DayOfYear"]), Isr["Temp"], .75)
    range_it = list(range(1, 11))
    losses = np.zeros((10,))
    for k in tqdm(range_it):
        model = PolynomialFitting(k)
        model.fit(train_x.to_numpy(), train_y.to_numpy())
        losses[k - 1] = np.round(model.loss(test_x.to_numpy(), test_y.to_numpy()), 2)
        print(f"Loss for {k} degree is: {losses[k - 1]}")
    plt.bar(range_it, losses)
    plt.title("Mean Loss with Different Polynomial Degree Of Fitting")
    plt.xlabel("Degree")
    plt.ylabel("Loss")
    plt.show()
    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(6)
    model.fit(Isr["DayOfYear"].to_numpy(), Isr["Temp"].to_numpy())
    Jdn = Xy[Xy["Country"] == "Jordan"]
    Saf = Xy[Xy["Country"] == "South Africa"]
    Net = Xy[Xy["Country"] == "The Netherlands"]
    loss_Jdn = np.round(model.loss(Jdn["DayOfYear"].to_numpy(), Jdn["Temp"].to_numpy()), 2)
    loss_Saf = np.round(model.loss(Saf["DayOfYear"].to_numpy(), Saf["Temp"].to_numpy()), 2)
    loss_Net = np.round(model.loss(Net["DayOfYear"].to_numpy(), Net["Temp"].to_numpy()), 2)
    plt.bar(["Jordan", "South Africa", "The Netherlands"], [loss_Jdn, loss_Saf, loss_Net])
    plt.title("Average Loss on different countries, by a model fitted on Israel")
    plt.xlabel("Country")
    plt.ylabel("Loss")
    plt.show()
