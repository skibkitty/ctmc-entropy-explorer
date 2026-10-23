"""
Parallel tracks model: a 6-state CTMC with a closed-form exact entropy
production rate. Flagship demo asset.
"""

import numpy as np

# States alternate between the two tracks: even states belong to track 1,
# odd states to track 2.
METASTATE_GROUPS = {"track1": [0, 2, 4], "track2": [1, 3, 5]}


def generate_rate_matrix_for_parallel_tracks(alpha, beta, u_one, w_one, u_two, w_two):
    """
    Generate rate matrix for parallel tracks model.

    Orientation convention: element (i, j) is the rate from state j → state i,
    i.e. the **column** is the starting state and the **row** is the ending
    state (the j-th column sums to zero). Some cited papers use the opposite
    convention (row = start, column = end); this implementation follows the
    column-start convention used throughout this project.

    Args:
        alpha, beta: Transition rates between tracks
        u_one, w_one: Forward and backward rates on track 1
        u_two, w_two: Forward and backward rates on track 2

    Returns:
        6x6 rate matrix for the parallel tracks system
    """
    rate_matrix = np.array([
        [-(u_one + w_one + beta),   alpha,          w_one,   0.0,   u_one,   0.0],
        [  beta, -(u_two + w_two + alpha),   0.0,   w_two,   0.0,   u_two],
        [ u_one,   0.0, -(u_one + w_one + beta),   alpha,   w_one,   0.0],
        [  0.0,   u_two,   beta, -(u_two + w_two + alpha),   0.0,   w_two],
        [  w_one,   0.0,  u_one,   0.0, -(u_one + w_one + beta),   alpha],
        [  0.0,   w_two,   0.0,   u_two,   beta, -(u_two + w_two + alpha)],
    ])

    return rate_matrix


def get_true_EPR_for_parallel_tracks(alpha, beta, u_one, w_one, u_two, w_two):
    """
    Calculate the exact entropy production rate for the parallel tracks model.

    This analytical result can be used to validate numerical estimators.

    Args:
        alpha, beta: Transition rates between tracks
        u_one, w_one: Forward and backward rates on track 1
        u_two, w_two: Forward and backward rates on track 2

    Returns:
        Exact entropy production rate
    """
    term1 = u_one * alpha * np.log(u_one / w_one)
    term2 = w_one * alpha * np.log(w_one / u_one)
    term3 = u_two * beta * np.log(u_two / w_two)
    term4 = w_two * beta * np.log(w_two / u_two)

    return (term1 + term2 + term3 + term4) / (alpha + beta)