""" 
Scripts repo for assignment 1
"""

from scipy.special import factorial

from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import sys
import os


# Helper function to check if computed gradient is reasonable...
def neg_likelihood_pois(beta, x, y):
    exp_x = np.clip(np.exp(beta) ** x, a_min=np.exp(-10), a_max=np.exp(10))
    nll = -beta * np.dot(x, y) + np.sum(exp_x) + np.sum(np.log(factorial(y)))

    return nll


def get_poisson_gradient(beta, x, y, lmbd):
    """
    In question 2.3, we computed the gradient for the negative log-likelihood

    x and y are vectors of size n.
    """
    # clip vector so we don't get 0 or inf.
    exp_x = np.clip(np.exp(beta) ** x, a_min=np.exp(-10), a_max=np.exp(10))
    # for debugging
    term1 = np.dot(x, exp_x)
    term2 = np.dot(x, y)

    if lmbd == None:
        gradient = term1 - term2
    else:
        ridge_regularizer = lmbd * beta
        gradient = term1 - term2 + ridge_regularizer

    return gradient


def get_function_value(beta, x, y):
    return neg_likelihood_pois(beta, x, y)


def gradient_descent(beta, x, y, lmbd=None):
    """Vanilla Gradient Descent

    Args:
        x (np array): x vector
        y (np array): y vector
        lmbd (int): lambda for ridge regularization. If lmbd provided, it is using ridge regularizer.

    Returns:
        _type_: Last gradient, function value, and optimal beta
    """

    # message to indicate ridge
    if lmbd != None:
        print("using ridge regularization with lambda: " + str(lmbd))

    max_evals = 150
    step_size = 1e-3
    epsilon = 10**-6

    # weights
    beta_old = beta  # initial beta is the first beta we call the function with.
    beta_new = None

    # gradients
    # initial gradient is computed with the first beta.
    gradient_old = get_poisson_gradient(beta_old, x, y, lmbd)
    gradient_new = None

    num_evals = 0
    list_of_function_values = {}  # list of function values to plot training progress.
    while True:
        # gradient descent step.
        beta_new = beta_old - step_size * gradient_old
        gradient_new = get_poisson_gradient(beta_new, x, y, lmbd)

        # for every 5th, get and save function value
        if num_evals % 1 == 0:
            list_of_function_values[beta_new] = get_function_value(beta_new, x, y)

        # Break condition: if gradient shrinks smaller than defined epsilon.
        # or if we've reached max evals.
        num_evals += 1
        if num_evals == max_evals:
            sys.exit("Reached max allowed evals.")
        if np.abs(gradient_new) <= epsilon:
            break

        # save previous values
        gradient_old = gradient_new
        beta_old = beta_new

    # return information
    return gradient_new, beta_new, list_of_function_values


def fit(x, y, lmbd=None):
    # set initial "guess" for weights. here, we set it to 0
    beta_guess = 0

    # pass lmbd if ridge regularized
    last_gradient, optimal_beta, list_of_function_values = gradient_descent(
        beta_guess, x, y, lmbd
    )
    return last_gradient, optimal_beta, list_of_function_values


def random_sample(beta, n):
    """
    Our random samples are a function of beta.
    """

    # make random samples of x and y.
    # random x drawn from standard normal distribution
    # random y drawn from poisson with theta = beta*xi
    x = np.random.randn(n)
    y = np.random.poisson(lam=np.exp(beta) ** x)

    return x, y


def main():
    os.chdir("/Users/et/Documents/WCM/courses/FODS/assignments/fods_2023_assignments")
    print(os.getcwd())

    # set seed
    np.random.seed(10)

    sample_size = 20

    x, y = random_sample(beta=3, n=sample_size)
    print(x)
    print(y)

    ################ Original Data ################
    plt.figure()
    plt.scatter(x, y, marker="o", color="blue", label="original")
    plt.xlabel("x value")
    plt.ylabel("y value")
    filename = Path("assignment_1", "figures", "data_seed_10.pdf")
    print("Saving to", filename)
    plt.savefig(filename)

    ################ modified poisson ################
    # fit unregularized modified poisson:
    mp_gradient, mp_optimal_beta, mp_list_of_function_values = fit(x, y)
    print("optimal beta: " + str(mp_optimal_beta))
    print("final gradient: " + str(mp_gradient))

    # plot with fitted model
    x_mp, y_mp = random_sample(beta=mp_optimal_beta, n=sample_size)

    # plot data
    plt.figure()
    plt.scatter(x, y, marker="o", color="blue", label="original")
    # plot a second layer
    plt.scatter(x_mp, y_mp, marker="*", color="red", label="modified poisson")
    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.legend()
    filename = Path("assignment_1", "figures", "modified_poisson_fit.pdf")
    print("Saving to", filename)
    plt.savefig(filename)

    ################ ridge modified poisson ################
    (
        rmp_gradient_lambda_1,
        rmp_optimal_beta_lambda_1,
        rmp_list_of_function_values_lambda_1,
    ) = fit(x, y, lmbd=1)
    (
        rmp_gradient_lambda_100,
        rmp_optimal_beta_lambda_100,
        rmp_list_of_function_values_lambda_100,
    ) = fit(x, y, lmbd=100)
    (
        rmp_gradient_lambda_500,
        rmp_optimal_beta_lambda_500,
        rmp_list_of_function_values_lambda_500,
    ) = fit(x, y, lmbd=500)

    # plot with fitted model
    x_rmp_lambda_1, y_rmp_lambda_1 = random_sample(
        beta=rmp_optimal_beta_lambda_1, n=sample_size
    )
    x_rmp_lambda_100, y_rmp_lambda_100 = random_sample(
        beta=rmp_optimal_beta_lambda_100, n=sample_size
    )
    x_rmp_lambda_500, y_rmp_lambda_500 = random_sample(
        beta=rmp_optimal_beta_lambda_500, n=sample_size
    )

    # plot data
    plt.figure()
    plt.scatter(x, y, marker="o", c="blue", label="original")
    # plot a second layer
    plt.scatter(
        x_rmp_lambda_1,
        y_rmp_lambda_1,
        marker="*",
        c="red",
        label="ridge modified poisson lambda = 1",
    )
    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.legend()

    filename = Path(
        "assignment_1",
        "figures",
        "ridge_modified_poisson_fit_lmbd_1.pdf",
    )
    print("Saving to", filename)
    plt.savefig(filename)

    plt.figure()
    plt.scatter(x, y, marker="o", c="blue", label="original")
    # plot a second layer
    plt.scatter(
        x_rmp_lambda_100,
        y_rmp_lambda_100,
        marker="*",
        c="red",
        label="ridge modified poisson lambda = 100",
    )
    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.legend()

    filename = Path(
        "assignment_1",
        "figures",
        "ridge_modified_poisson_fit_lmbd_100.pdf",
    )
    print("Saving to", filename)
    plt.savefig(filename)

    plt.figure()
    plt.scatter(x, y, marker="o", c="blue", label="original")
    # plot a second layer
    plt.scatter(
        x_rmp_lambda_500,
        y_rmp_lambda_500,
        marker="*",
        c="red",
        label="ridge modified poisson lambda = 500",
    )
    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.legend()
    filename = Path(
        "assignment_1",
        "figures",
        "ridge_modified_poisson_fit_lmbd_500.pdf",
    )
    print("Saving to", filename)
    plt.savefig(filename)

    #################### Plot Learning from training ######################
    plt.figure()
    plt.scatter(
        mp_list_of_function_values.keys(),
        mp_list_of_function_values.values(),
        marker="o",
        c="blue",
        label="modified poisson learning",
    )
    plt.xlabel("beta")
    plt.ylabel("NLL")
    plt.legend()
    filename = Path(
        "assignment_1",
        "figures",
        "modified_poisson_learning.pdf",
    )
    print("Saving to", filename)
    plt.savefig(filename)

    plt.figure()
    plt.scatter(
        rmp_list_of_function_values_lambda_100.keys(),
        rmp_list_of_function_values_lambda_100.values(),
        marker="o",
        c="blue",
        label="modified poisson with ridge learning lambda = 100",
    )
    plt.xlabel("beta")
    plt.ylabel("NLL")
    plt.legend()
    filename = Path(
        "assignment_1",
        "figures",
        "ridge_modified_poisson_learning_lambda_100.pdf",
    )
    print("Saving to", filename)
    plt.savefig(filename)


if __name__ == "__main__":
    main()
