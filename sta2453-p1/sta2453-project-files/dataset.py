import pandas as pd
import numpy as np

from utils.black_scholes import black_scholes_put
from scipy.stats import truncnorm

# Domain boundaries
S_bound = (0.0, 200.0)
K_bound = (50.0, 150.0)
T_bound = (0.0, 5.0)
r_bound = (0.001, 0.05)
sigma_bound = (0.05, 1.5)


def generate_data(n):
    # generate standardized normal rvs with range (0,1)
    rv = get_truncated_normal()
    return rv.rvs(size=(n, 5))


def get_truncated_normal(mean=0, sd=1, low=0, upp=1):
    """"
    helper function to truncate normal distribution
    default: standardized normal with range (0,1)
    """
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def generate_black_scholes_put_data(n):
    x = generate_data(n)

    S_delta = S_bound[1] - S_bound[0]
    K_delta = K_bound[1] - K_bound[0]
    T_delta = T_bound[1] - T_bound[0]
    r_delta = r_bound[1] - r_bound[0]
    sigma_delta = sigma_bound[1] - sigma_bound[0]

    deltas = np.array([S_delta, K_delta, T_delta, r_delta, sigma_delta])
    l_bounds = np.array([S_bound[0], K_bound[0], T_bound[0], r_bound[0], sigma_bound[0]])

    x = x * deltas + l_bounds
    # use black scholes analytic pricer to predict the value
    y = black_scholes_put(S=x[:, 0], K=x[:, 1], T=x[:, 2], r=x[:, 3], sigma=x[:, 4]).reshape(-1, 1)

    return np.append(x, y, axis=1)


def main():
    """
    Generate Black Scholes put data points for training, validation, testing
    """

    columns = ["S", "K", "T", "r", "sigma", "value"]

    train = generate_black_scholes_put_data(20_000)
    train_df = pd.DataFrame(train, columns=columns)
    train_df.to_csv("training.csv")

    validation = generate_black_scholes_put_data(3_000)
    validation_df = pd.DataFrame(validation, columns=columns)
    validation_df.to_csv("validation.csv")

    test = generate_black_scholes_put_data(3_000)
    test_df = pd.DataFrame(test, columns=columns)
    test_df.to_csv("testing.csv")


if __name__ == "__main__":
    main()
