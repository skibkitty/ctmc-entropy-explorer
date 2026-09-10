"""
Entropy production estimators.

Port of the research module's estimators. All functions are pure: they return
values, never print, and raise ``ValueError`` on invalid input instead of
degrading silently.
"""

import math
from collections import Counter
from itertools import groupby

import numpy as np

from .simulation import count_transitions_fast, find_snippet_boundaries_fast

# ============================================================================
# k-th order estimator
# ============================================================================

def kth_order_estimator(trajectory, k, total_simulation_time):
    """
    Estimate entropy production using k-th order transition statistics.

    Compares frequencies of k+1-length state sequences to their time-reversed
    versions. For Markovian systems this gives exact results for all k (given
    sufficient statistics).

    Mathematical basis:
        EPR ≈ (1/kτ) Σ P(s₁...sₖ₊₁) ln[P(s₁...sₖ₊₁) / P(sₖ₊₁...s₁)]
    where τ is the mean time between transitions.

    Args:
        trajectory: List of visited states
        k: Order of estimator (k=1 is the Markov estimator)
        total_simulation_time: Total simulation time

    Returns:
        Estimated entropy production rate
    """
    if k < 1:
        raise ValueError("k must be a positive integer (k >= 1)")
    if total_simulation_time < 0:
        raise ValueError("total_simulation_time must be non-negative")

    if len(trajectory) < k + 1:
        return 0.0

    # Extract all (k+1)-length state sequences
    snippets = [tuple(trajectory[i:i + k + 1]) for i in range(len(trajectory) - k)]
    counts = Counter(snippets)

    num_snippets = sum(counts.values())
    if num_snippets == 0:
        return 0.0

    probabilities = {snippet: count / num_snippets for snippet, count in counts.items()}

    # Calculate entropy production estimate
    entropy_estimate = 0.0
    for snippet, prob in probabilities.items():
        reversed_snippet = tuple(reversed(snippet))
        prob_reversed = probabilities.get(reversed_snippet, 0.0)

        if prob_reversed > 0:
            entropy_estimate += prob * math.log(prob / prob_reversed)

    # Normalize by mean transition time and order
    mean_time_between_transitions = total_simulation_time / (num_snippets + k - 1)

    if mean_time_between_transitions > 0:
        return entropy_estimate / mean_time_between_transitions / k
    else:
        return 0.0


# ============================================================================
# Repeated transitions estimator
# ============================================================================

def count_repeated_transition_patterns(trajectory, state1, state2):
    """
    Count conditional transition patterns between two states.

    This is the pattern-counting helper for the Harunari repeated-transitions
    estimator (not a waiting/residence-time analysis). It counts four specific
    patterns:
    1. state1→state2 ... (no state2→state1) ... state1→state2
    2. state1→state2 ... (no state1→state2) ... state2→state1
    3. state2→state1 ... (no state1→state2) ... state2→state1
    4. state2→state1 ... (no state2→state1) ... state1→state2

    Args:
        trajectory: List of states
        state1: First state of interest
        state2: Second state of interest

    Returns:
        Dictionary with counts for each pattern
    """
    transitions = [(trajectory[i], trajectory[i + 1])
                   for i in range(len(trajectory) - 1)]
    length = len(transitions)

    pattern1 = pattern2 = pattern3 = pattern4 = 0
    forward = (state1, state2)
    backward = (state2, state1)

    for i1 in range(length - 1):
        # Pattern 1: forward ... (no backward) ... forward
        if transitions[i1] == forward:
            i2 = i1 + 1
            forbidden = False
            while i2 < length and transitions[i2] != forward:
                if transitions[i2] == backward:
                    forbidden = True
                    break
                i2 += 1
            if i2 < length and not forbidden:
                pattern1 += 1

        # Pattern 2: forward ... (no forward) ... backward
        if transitions[i1] == forward:
            i2 = i1 + 1
            forbidden = False
            while i2 < length and transitions[i2] != backward:
                if transitions[i2] == forward:
                    forbidden = True
                    break
                i2 += 1
            if i2 < length and not forbidden:
                pattern2 += 1

        # Pattern 3: backward ... (no forward) ... backward
        if transitions[i1] == backward:
            i2 = i1 + 1
            forbidden = False
            while i2 < length and transitions[i2] != backward:
                if transitions[i2] == forward:
                    forbidden = True
                    break
                i2 += 1
            if i2 < length and not forbidden:
                pattern3 += 1

        # Pattern 4: backward ... (no backward) ... forward
        if transitions[i1] == backward:
            i2 = i1 + 1
            forbidden = False
            while i2 < length and transitions[i2] != forward:
                if transitions[i2] == backward:
                    forbidden = True
                    break
                i2 += 1
            if i2 < length and not forbidden:
                pattern4 += 1

    return {
        f"{forward} ...(!{backward})... {forward}": pattern1,
        f"{forward} ...(!{forward})... {backward}": pattern2,
        f"{backward} ...(!{forward})... {backward}": pattern3,
        f"{backward} ...(!{backward})... {forward}": pattern4
    }


def repeated_transitions_estimator(trajectory, start, end, total_simulation_time,
                                   state_mapping=None):
    """
    Estimate entropy production using the repeated transitions method.

    This estimator is exact for unicyclic systems (e.g. single-file diffusion).
    It uses conditional transition frequencies to estimate the entropy
    production rate between two coarse-grained states.

    Mathematical basis:
        EPR ≈ R × (f₊ - f₋) × ln(f₊|₊ / f₋|₋)
    where R is the transition rate, f₊/f₋ are forward/backward frequencies,
    and f₊|₊, f₋|₋ are conditional forward/backward frequencies.

    Args:
        trajectory: List of states
        start: Starting state for analysis
        end: Ending state for analysis
        total_simulation_time: Total simulation time
        state_mapping: Optional dict to map states to coarse-grained states

    Returns:
        Estimated entropy production rate
    """
    if state_mapping is not None:
        converted_trajectory = [state_mapping.get(state, state) for state in trajectory]
    else:
        converted_trajectory = trajectory

    # Remove consecutive duplicates
    coarse_trajectory = [key for key, _ in groupby(converted_trajectory)]

    # Count first-order transitions
    first_order_transitions = [
        (coarse_trajectory[i], coarse_trajectory[i + 1])
        for i in range(len(coarse_trajectory) - 1)
    ]
    first_order_counts = Counter(first_order_transitions)
    counts_plus = first_order_counts.get((start, end), 0)
    counts_minus = first_order_counts.get((end, start), 0)

    # Count conditional transitions
    conditional_transitions = count_repeated_transition_patterns(coarse_trajectory, start, end)
    patterns = list(conditional_transitions.values())
    counts_plus_given_plus = patterns[0] if len(patterns) > 0 else 0
    counts_minus_given_plus = patterns[1] if len(patterns) > 1 else 0
    counts_minus_given_minus = patterns[2] if len(patterns) > 2 else 0
    counts_plus_given_minus = patterns[3] if len(patterns) > 3 else 0

    # Calculate frequencies
    total_transitions = counts_plus + counts_minus
    if total_transitions == 0:
        return 0.0

    freq_plus = counts_plus / total_transitions
    freq_minus = counts_minus / total_transitions

    plus_total = counts_plus_given_plus + counts_minus_given_plus
    minus_total = counts_plus_given_minus + counts_minus_given_minus

    if plus_total == 0 or minus_total == 0:
        return 0.0

    freq_plus_given_plus = counts_plus_given_plus / plus_total
    freq_minus_given_minus = counts_minus_given_minus / minus_total

    if freq_minus_given_minus > 0 and total_simulation_time > 0:
        log_ratio = math.log(freq_plus_given_plus / freq_minus_given_minus)
        transition_rate = total_transitions / total_simulation_time
        return transition_rate * (freq_plus - freq_minus) * log_ratio
    else:
        return 0.0


# ============================================================================
# Thermodynamic uncertainty relation estimator
# ============================================================================

def thermodynamic_uncertainty_relation_estimator(trajectory, times,
                                                snippet_time_length=None,
                                                state_mapping=None):
    """
    Estimate entropy production using the thermodynamic uncertainty relation (TUR).

    The TUR estimator divides the trajectory into equal-time windows and
    computes the mean and variance of net transitions. The entropy production
    rate is bounded by:

        EPR >= 2 / (uncertainty^2 * T)

    This provides a lower bound on entropy production that is particularly
    useful for systems with complex dynamics.

    Args:
        trajectory: List of state indices
        times: List of corresponding time points
        snippet_time_length: Duration of each time window
        state_mapping: Dict mapping state indices to coarse-grained states

    Returns:
        Estimated entropy production rate (lower bound)
    """
    if state_mapping is None:
        state_mapping = {0: "A", 1: "A", 2: "B", 3: "B", 4: "C", 5: "C"}

    if snippet_time_length is None:
        raise ValueError("snippet_time_length must be specified")

    traj_array = np.array(trajectory, dtype=np.int32)
    times_array = np.array(times, dtype=np.float64)

    max_state = max(state_mapping.keys())
    state_lookup = np.array([''] * (max_state + 1), dtype='U1')
    for k, v in state_mapping.items():
        state_lookup[k] = v
    converted = state_lookup[traj_array]

    snippet_boundaries = find_snippet_boundaries_fast(times_array, snippet_time_length)
    num_snippets = len(snippet_boundaries)

    net_transitions = np.zeros(num_snippets, dtype=np.float64)
    time_diffs = np.zeros(num_snippets, dtype=np.float64)

    for i, (start, end) in enumerate(snippet_boundaries):
        snippet = converted[start:end]

        if len(snippet) < 2:
            time_diffs[i] = times_array[min(end - 1, len(times_array) - 1)] - times_array[start]
            continue

        ab_count, ba_count, bc_count, cb_count, ca_count, ac_count = \
            count_transitions_fast(snippet)

        time_diff = times_array[end - 1] - times_array[start]
        time_diffs[i] = time_diff

        if time_diff == 0:
            continue

        # Calculate net transitions (signed sum based on cycle direction)
        d_ab = d_bc = d_ca = 1   # Forward cycle
        d_ba = d_cb = d_ac = -1  # Backward cycle

        net_count = (d_ab * ab_count + d_ba * ba_count + d_ac * ac_count +
                     d_ca * ca_count + d_bc * bc_count + d_cb * cb_count)

        net_transitions[i] = net_count

    mean_net = np.mean(net_transitions)

    if mean_net == 0:
        return float('inf')

    mean_net_sq = np.mean(net_transitions ** 2)
    variance = mean_net_sq - mean_net ** 2
    uncertainty_squared = variance / (mean_net ** 2)

    if uncertainty_squared <= 0:
        return float('inf')

    return 2 / (uncertainty_squared * snippet_time_length)