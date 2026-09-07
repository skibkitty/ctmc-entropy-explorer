"""
Continuous-Time Markov Chain Simulator with Entropy Production Analysis

This module provides tools for simulating continuous-time Markov chains and computing
entropy production rates using various estimators. Optimized with Numba JIT compilation
for high-performance trajectory generation.

Key Features:
- Efficient Continuous-Time Markov Chain simulation with coarse-grained metastate tracking
- Multiple entropy production estimators (k-th order, repeated transitions, TUR)
- Trajectory coarse-graining and analysis
- Rate matrix utilities for common systems
"""

import numpy as np
import time
import os
import pickle
from numba import jit, types
from numba.typed import Dict
from collections import Counter, defaultdict
import math
from itertools import groupby
import matplotlib.pyplot as plt


# ============================================================================
# Core Simulation Functions (JIT-Compiled)
# ============================================================================

@jit(nopython=True)
def simulate_trajectory_core(rate_matrix, state_to_metastate_array, max_length, 
                            initial_state, exit_rates, transition_matrices):
    """
    Core simulation loop for continuous-time Markov chain trajectories.
    
    Uses the Gillespie algorithm to generate exact stochastic trajectories:
    1. Sample waiting time exponentially based on current state's exit rate
    2. Sample next state according to transition probabilities
    3. Track metastate transitions and residence times
    
    Args:
        rate_matrix: Square matrix of transition rates between states
        state_to_metastate_array: Maps each state index to its metastate
        max_length: Maximum number of transitions to simulate
        initial_state: Starting state index
        exit_rates: Pre-computed exit rates for each state
        transition_matrices: Pre-computed transition probability matrices
        
    Returns:
        trajectory: Array of visited states
        times: Array of arrival times
        metastate_changes: Indices of metastates visited
        waiting_times_raw: Time spent in each metastate before transition
    """
    n_states = rate_matrix.shape[0]
    
    # Pre-allocate trajectory storage
    trajectory = np.zeros(max_length, dtype=np.int32)
    times = np.zeros(max_length, dtype=np.float64)
    
    # Initialize simulation state
    current_state = initial_state
    trajectory[0] = current_state
    times[0] = 0.0
    time_elapsed = 0.0
    current_length = 1
    
    # Track metastate transitions
    current_metastate = state_to_metastate_array[current_state]
    time_entered_current_metastate = 0.0
    
    # Storage for metastate residence times
    metastate_changes = np.zeros(max_length, dtype=np.int32)
    waiting_times_raw = np.zeros(max_length, dtype=np.float64)
    num_metastate_records = 0
    
    # Main simulation loop (Gillespie algorithm)
    while current_length < max_length:
        exit_rate = exit_rates[current_state]
        
        if exit_rate <= 0:
            # exit_rate should never be negative nor zero (the diagonal of rate matrix is written as the negative escape rate)
            print("Warning: Reached state with negative exit rate.")
            break
        
        # Sample waiting time using inverse transform sampling
        waiting_time = -np.log(np.random.random()) / exit_rate
        time_elapsed += waiting_time
        
        # Sample next state from transition probabilities
        rand_val = np.random.random()
        cumsum = 0.0
        next_state = current_state  # Fallback
        
        for j in range(n_states):
            if j != current_state:
                cumsum += transition_matrices[current_state, j]
                if rand_val <= cumsum:
                    next_state = j
                    break
        
        # Check for metastate transition
        next_metastate = state_to_metastate_array[next_state]
        
        if next_metastate != current_metastate:
            # Record time spent in current metastate (aka coarse-grained state)
            time_in_current_metastate = time_elapsed - time_entered_current_metastate
            metastate_changes[num_metastate_records] = current_metastate
            waiting_times_raw[num_metastate_records] = time_in_current_metastate
            num_metastate_records += 1
            time_entered_current_metastate = time_elapsed
        
        # Update trajectory
        current_state = next_state
        current_metastate = next_metastate
        trajectory[current_length] = current_state
        times[current_length] = time_elapsed
        current_length += 1
    
    return (trajectory[:current_length], times[:current_length], 
            metastate_changes[:num_metastate_records], 
            waiting_times_raw[:num_metastate_records])


def precompute_transition_data(rate_matrix):
    """
    Pre-compute exit rates and transition probabilities for faster simulation.
    
    For each state i:
    - Exit rate = -rate_matrix[i,i] (negative of diagonal element)
    - Transition probabilities = off-diagonal rates / exit rate
    
    Args:
        rate_matrix: Square matrix of transition rates
        
    Returns:
        exit_rates: Array of exit rates for each state
        transition_matrices: Matrix of transition probabilities
    """
    n_states = rate_matrix.shape[0]
    exit_rates = np.zeros(n_states)
    transition_matrices = np.zeros((n_states, n_states))
    
    for i in range(n_states):
        exit_rate = -rate_matrix[i, i]
        exit_rates[i] = exit_rate
        
        if exit_rate > 0:
            # Normalize off-diagonal rates to get transition probabilities
            transition_rates = rate_matrix[:, i].copy()
            transition_rates[i] = 0
            transition_matrices[i, :] = transition_rates / exit_rate
    
    return exit_rates, transition_matrices


def process_metastate_waiting_times(metastate_changes, waiting_times_raw, metastate_groups):
    """
    Process raw metastate data into structured format.
    
    Aggregates residence times by metastate and computes statistics.
    
    Args:
        metastate_changes: Array of metastate indices visited
        waiting_times_raw: Array of residence times
        metastate_groups: Dictionary mapping metastate names to state lists
        
    Returns:
        waiting_times_by_metastate: Lists of residence times per metastate
        time_in_metastates: Total time spent in each metastate
        metastate_transition_count: Number of metastate transitions
    """
    metastate_names = list(metastate_groups.keys())
    
    # Initialize storage
    waiting_times_by_metastate = {name: [] for name in metastate_names}
    
    # Aggregate residence times by metastate
    for i in range(len(metastate_changes)):
        metastate_idx = metastate_changes[i]
        waiting_time = waiting_times_raw[i]
        
        if metastate_idx < len(metastate_names) and waiting_time > 0:
            metastate_name = metastate_names[metastate_idx]
            waiting_times_by_metastate[metastate_name].append(waiting_time)
    
    # Compute total times
    time_in_metastates = {
        metastate: np.sum(waiting_list) 
        for metastate, waiting_list in waiting_times_by_metastate.items()
    }
    
    return waiting_times_by_metastate, time_in_metastates, len(metastate_changes)


# ============================================================================
# Main Simulation Interface
# ============================================================================

def simulate_single_trajectory(rate_matrix, metastate_groups, max_length=1000, 
                              initial_state=0, save_path=None):
    """
    Simulate a continuous-time Markov chain trajectory with metastate tracking.
    
    This is the main user-facing simulation function. It handles validation,
    pre-computation, and post-processing of results.
    
    Args:
        rate_matrix: Square matrix where element (i,j) is the rate from j to i
        metastate_groups: Dict mapping metastate names to lists of state indices
        max_length: Maximum trajectory length (number of transitions)
        initial_state: Starting state index
        save_path: Optional path to save results as pickle file
        
    Returns:
        trajectory: List of visited states
        times: List of arrival times
        waiting_times_by_metastate: Dict of residence time lists per metastate
        time_in_metastates: Dict of total times per metastate
        metastate_transition_count: Number of metastate transitions
    """
    # Convert and validate inputs
    rate_matrix = np.array(rate_matrix, dtype=np.float64)
    n_states = rate_matrix.shape[0]
    
    if rate_matrix.shape[0] != rate_matrix.shape[1]:
        raise ValueError("Rate matrix must be square")
    
    # Validate rate matrix normalization
    for i in range(n_states):
        off_diagonal_sum = np.sum(rate_matrix[:, i]) - rate_matrix[i, i]
        expected_diagonal = -off_diagonal_sum
        if not np.isclose(rate_matrix[i, i], expected_diagonal, rtol=1e-10):
            print(f"Warning: Rate matrix may not be properly normalized at state {i}")
    
    # Create state to metastate mapping
    state_to_metastate = {}
    metastate_names = list(metastate_groups.keys())
    
    for metastate_idx, (metastate_name, state_list) in enumerate(metastate_groups.items()):
        for state_idx in state_list:
            if state_idx >= n_states:
                raise ValueError(f"State index {state_idx} exceeds matrix size")
            state_to_metastate[state_idx] = metastate_idx
    
    state_to_metastate_array = np.array([
        state_to_metastate.get(i, 0) for i in range(n_states)
    ])
    
    # Pre-compute transition data for performance
    print("Pre-computing transition data...")
    exit_rates, transition_matrices = precompute_transition_data(rate_matrix)
    
    print(f"Starting simulation with {n_states} states, {len(metastate_names)} metastates")
    print(f"Initial state: {initial_state}")
    
    # Run simulation
    start_time = time.time()
    trajectory, times, metastate_changes, waiting_times_raw = simulate_trajectory_core(
        rate_matrix, state_to_metastate_array, max_length, initial_state,
        exit_rates, transition_matrices
    )
    simulation_time = time.time() - start_time
    print(f"Core simulation completed in {simulation_time:.2f} seconds")
    
    # Process results
    print("Processing metastate transitions...")
    analysis_start = time.time()
    waiting_times_by_metastate, time_in_metastates, metastate_transition_count = \
        process_metastate_waiting_times(metastate_changes, waiting_times_raw, metastate_groups)
    analysis_time = time.time() - analysis_start
    print(f"Analysis completed in {analysis_time:.2f} seconds")
    
    # Print summary
    final_time = times[-1] if len(times) > 0 else 0
    print(f"\nSimulation complete. Total time: {final_time:.2f}")
    print(f"Trajectory length: {len(trajectory)}")
    
    for metastate_name in metastate_names:
        total_time = time_in_metastates.get(metastate_name, 0)
        num_visits = len(waiting_times_by_metastate.get(metastate_name, []))
        avg_waiting_time = total_time / num_visits if num_visits > 0 else 0
        print(f"Metastate {metastate_name}: {total_time:.2f} total time, "
              f"{num_visits} visits, {avg_waiting_time:.2f} avg per visit")
        
        if num_visits > 0:
            sample_times = waiting_times_by_metastate[metastate_name][:5]
            print(f"  Sample waiting times: {[f'{t:.3f}' for t in sample_times]}")
    
    print(f"Total metastate transitions: {metastate_transition_count}")
    print(f"Total execution time: {simulation_time + analysis_time:.2f} seconds")
    
    # Save results if requested
    if save_path:
        simulation_data = {
            'trajectory': trajectory.tolist(),
            'times': times.tolist(),
            'waiting_times_by_metastate': waiting_times_by_metastate,
            'metastate_transition_count': metastate_transition_count,
            'parameters': {
                'rate_matrix': rate_matrix.tolist(),
                'metastate_groups': metastate_groups,
                'max_length': max_length,
                'initial_state': initial_state,
                'final_time': final_time
            },
            'performance': {
                'simulation_time': simulation_time,
                'analysis_time': analysis_time,
                'total_time': simulation_time + analysis_time
            }
        }
        
        save_directory = os.path.dirname(save_path)
        if save_directory and not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        with open(save_path, 'wb') as f:
            pickle.dump(simulation_data, f)
        print(f"Data saved to {save_path}")
    
    return trajectory, times, waiting_times_by_metastate, time_in_metastates, \
           metastate_transition_count


# ============================================================================
# Entropy Production Estimators
# ============================================================================

def kth_order_estimator(trajectory, k, total_simulation_time, print_counts=False):
    """
    Estimate entropy production using k-th order transition statistics.
    
    This estimator compares frequencies of k+1-length state sequences to their
    time-reversed versions. For Markovian systems, this should give exact results
    for all k (given sufficient statistics).
    
    Mathematical basis:
        EPR ≈ (1/kτ) Σ P(s₁...sₖ₊₁) ln[P(s₁...sₖ₊₁) / P(sₖ₊₁...s₁)]
    where τ is the mean time between transitions.
    
    Args:
        trajectory: List of visited states
        k: Order of estimator (k=1 is the Markov estimator)
        total_simulation_time: Total simulation time
        print_counts: Whether to print sequence counts for debugging
        
    Returns:
        Estimated entropy production rate
    """
    if len(trajectory) < k + 1:
        return 0.0
    
    # Extract all (k+1)-length state sequences
    snippets = [tuple(trajectory[i:i+k+1]) for i in range(len(trajectory)-k)]
    counts = Counter(snippets)
    
    if print_counts:
        print(f"k={k}:")
        for snippet, count in counts.items():
            print(f"{snippet}: {count}")
        print()
    
    num_snippets = sum(counts.values())
    if num_snippets == 0:
        return 0.0
    
    # Compute probabilities
    probabilities = {snippet: count/num_snippets for snippet, count in counts.items()}
    
    # Calculate entropy production estimate
    entropy_estimate = 0.0
    for snippet, prob in probabilities.items():
        reversed_snippet = tuple(reversed(snippet))
        prob_reversed = probabilities.get(reversed_snippet, 0.0)
        
        # Only include bidirectional transitions
        if prob_reversed > 0:
            entropy_estimate += prob * math.log(prob / prob_reversed)
    
    # Normalize by mean transition time and order
    mean_time_between_transitions = total_simulation_time / (num_snippets + k - 1)
    
    if mean_time_between_transitions > 0 and k > 0:
        return entropy_estimate / mean_time_between_transitions / k
    else:
        return 0.0


def analyze_transitions(trajectory, state1, state2):
    """
    Analyze conditional transition patterns between two states.
    
    Counts four specific patterns for repeated transitions estimator:
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
    # Create all consecutive transition pairs
    transitions = [(trajectory[i], trajectory[i+1]) 
                   for i in range(len(trajectory)-1)]
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
                                   state_mapping=None, show_debug_prints=False):
    """
    Estimate entropy production using repeated transitions method.
    
    This estimator is exact for unicyclic systems (e.g., single-file diffusion).
    It uses conditional transition frequencies to estimate the entropy production
    rate between two coarse-grained states.
    
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
        show_debug_prints: Whether to print detailed statistics
        
    Returns:
        Estimated entropy production rate
    """
    # Apply coarse-graining if specified
    if state_mapping is not None:
        converted_trajectory = [state_mapping.get(state, state) for state in trajectory]
    else:
        converted_trajectory = trajectory
    
    # Remove consecutive duplicates
    coarse_trajectory = [key for key, _ in groupby(converted_trajectory)]
    
    if show_debug_prints:
        print(f"Original trajectory length: {len(trajectory)}")
        print(f"Coarse trajectory length: {len(coarse_trajectory)}")
        print(f"First 20 coarse states: {coarse_trajectory[:20]}")
    
    # Count first-order transitions
    first_order_transitions = [(coarse_trajectory[i], coarse_trajectory[i+1]) for i in range(len(coarse_trajectory)-1)]
    first_order_counts = Counter(first_order_transitions)
    counts_plus = first_order_counts.get((start, end), 0)
    counts_minus = first_order_counts.get((end, start), 0)
    
    # Count conditional transitions
    conditional_transitions = analyze_transitions(coarse_trajectory, start, end)
    patterns = list(conditional_transitions.values())
    counts_plus_given_plus = patterns[0] if len(patterns) > 0 else 0
    counts_minus_given_plus = patterns[1] if len(patterns) > 1 else 0
    counts_minus_given_minus = patterns[2] if len(patterns) > 2 else 0
    counts_plus_given_minus = patterns[3] if len(patterns) > 3 else 0
    
    if show_debug_prints:
        print(f"First order counts: {dict(first_order_counts)}")
        print(f"Conditional transitions: {conditional_transitions}")
        print(f"counts_plus: {counts_plus}")
        print(f"counts_minus: {counts_minus}")
        print(f"counts_plus_given_plus: {counts_plus_given_plus}")
        print(f"counts_minus_given_minus: {counts_minus_given_minus}")
        print(f"counts_plus_given_minus: {counts_plus_given_minus}")
        print(f"counts_minus_given_plus: {counts_minus_given_plus}")
    
    # Calculate frequencies
    total_transitions = counts_plus + counts_minus
    if total_transitions == 0:
        print("No transitions found between specified states")
        return 0.0
    
    freq_plus = counts_plus / total_transitions
    freq_minus = counts_minus / total_transitions
    
    # Calculate conditional frequencies
    plus_total = counts_plus_given_plus + counts_minus_given_plus
    minus_total = counts_plus_given_minus + counts_minus_given_minus
    
    if plus_total == 0 or minus_total == 0:
        print("Insufficient data for conditional frequency calculation")
        return 0.0
    
    freq_plus_given_plus = counts_plus_given_plus / plus_total
    freq_minus_given_minus = counts_minus_given_minus / minus_total
    
    if show_debug_prints:
        print(f"freq_plus: {freq_plus:.4f}")
        print(f"freq_minus: {freq_minus:.4f}")
        print(f"freq_plus_given_plus: {freq_plus_given_plus:.4f}")
        print(f"freq_minus_given_minus: {freq_minus_given_minus:.4f}")
    
    # Compute entropy production rate
    if freq_minus_given_minus > 0 and total_simulation_time > 0:
        log_ratio = math.log(freq_plus_given_plus / freq_minus_given_minus)
        transition_rate = total_transitions / total_simulation_time
        estimated_epr = transition_rate * (freq_plus - freq_minus) * log_ratio
    else:
        estimated_epr = 0.0
    
    if show_debug_prints:
        print(f"Repeated Transitions Estimate: {estimated_epr}")
        
        # Compare with k-th order estimates
        for k in range(1, 5):
            k_estimate = kth_order_estimator(coarse_trajectory, k, 
                                            total_simulation_time, print_counts=True)
            print(f"k={k} estimator: {k_estimate}")
    
    return estimated_epr


@jit(nopython=True)
def find_snippet_boundaries_fast(times_array, snippet_time_length):
    """
    Fast identification of time window boundaries for TUR estimator.
    
    Uses binary search to efficiently partition trajectory into equal-time snippets.
    
    Args:
        times_array: Array of transition times
        snippet_time_length: Duration of each time window
        
    Returns:
        List of (start_idx, end_idx) tuples for each snippet
    """
    boundaries = []
    current_time = times_array[0]
    start_idx = 0
    
    while start_idx < len(times_array):
        target_time = current_time + snippet_time_length
        
        # Binary search for end of time window
        end_idx = start_idx
        while end_idx < len(times_array) and times_array[end_idx] < target_time:
            end_idx += 1
        
        # Ensure non-empty window
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        
        boundaries.append((start_idx, min(end_idx, len(times_array))))
        
        start_idx = end_idx
        if start_idx < len(times_array):
            current_time = times_array[start_idx]
    
    return boundaries


@jit(nopython=True)
def count_transitions_fast(converted_snippet):
    """
    Fast transition counting for TUR estimator (Numba-optimized).
    
    Counts all directed transitions between coarse-grained states A, B, C.
    
    Args:
        converted_snippet: Array of coarse-grained state labels
        
    Returns:
        Tuple of (ab_count, ba_count, bc_count, cb_count, ca_count, ac_count)
    """
    if len(converted_snippet) < 2:
        return 0, 0, 0, 0, 0, 0
    
    # Remove consecutive duplicates to get coarse trajectory
    coarse_traj = [converted_snippet[0]]
    for i in range(1, len(converted_snippet)):
        if converted_snippet[i] != converted_snippet[i-1]:
            coarse_traj.append(converted_snippet[i])
    
    if len(coarse_traj) < 2:
        return 0, 0, 0, 0, 0, 0
    
    # Count all transitions
    ab_count = ba_count = bc_count = cb_count = ca_count = ac_count = 0
    
    for i in range(len(coarse_traj) - 1):
        curr_state = coarse_traj[i]
        next_state = coarse_traj[i + 1]
        
        if curr_state == 'A' and next_state == 'B':
            ab_count += 1
        elif curr_state == 'B' and next_state == 'A':
            ba_count += 1
        elif curr_state == 'B' and next_state == 'C':
            bc_count += 1
        elif curr_state == 'C' and next_state == 'B':
            cb_count += 1
        elif curr_state == 'C' and next_state == 'A':
            ca_count += 1
        elif curr_state == 'A' and next_state == 'C':
            ac_count += 1
    
    return ab_count, ba_count, bc_count, cb_count, ca_count, ac_count


def thermodynamic_uncertainty_relation_estimator(trajectory, times, 
                                                snippet_time_length=None, 
                                                state_mapping=None):
    """
    Estimate entropy production using thermodynamic uncertainty relation (TUR).
    
    The TUR estimator divides the trajectory into equal-time windows and computes
    the mean and variance of net transitions. The entropy production rate is
    bounded by:
        EPR >= 2 / (uncertainty^2 * T)
    
    This provides a lower bound on entropy production that's particularly useful
    for systems with complex dynamics.
    
    Args:
        trajectory: List of state indices
        times: List of corresponding time points
        snippet_time_length: Duration of each time window
        state_mapping: Dict mapping state indices to coarse-grained states
        
    Returns:
        Estimated entropy production rate (lower bound)
    """
    # Default coarse-graining for 3-metastate systems
    if state_mapping is None:
        state_mapping = {0: "A", 1: "A", 2: "B", 3: "B", 4: "C", 5: "C"}
    
    if snippet_time_length is None:
        raise ValueError("snippet_time_length must be specified")
    
    # Convert to numpy arrays for efficiency
    traj_array = np.array(trajectory, dtype=np.int32)
    times_array = np.array(times, dtype=np.float64)
    
    # Create fast lookup table for state mapping
    max_state = max(state_mapping.keys())
    state_lookup = np.array([''] * (max_state + 1), dtype='U1')
    for k, v in state_mapping.items():
        state_lookup[k] = v
    converted = state_lookup[traj_array]
    
    # Partition trajectory into time windows
    snippet_boundaries = find_snippet_boundaries_fast(times_array, snippet_time_length)
    num_snippets = len(snippet_boundaries)
    
    # Pre-allocate results
    net_transitions = np.zeros(num_snippets, dtype=np.float64)
    time_diffs = np.zeros(num_snippets, dtype=np.float64)
    
    # Process each time window
    for i, (start, end) in enumerate(snippet_boundaries):
        snippet = converted[start:end]
        
        if len(snippet) < 2:
            time_diffs[i] = times_array[min(end-1, len(times_array)-1)] - times_array[start]
            continue
        
        # Count transitions in this window
        ab_count, ba_count, bc_count, cb_count, ca_count, ac_count = \
            count_transitions_fast(snippet)
        
        time_diff = times_array[end-1] - times_array[start]
        time_diffs[i] = time_diff
        
        if time_diff == 0:
            continue
        
        # Calculate net transitions (signed sum based on cycle direction)
        d_ab = d_bc = d_ca = 1   # Forward cycle
        d_ba = d_cb = d_ac = -1  # Backward cycle
        
        net_count = (d_ab*ab_count + d_ba*ba_count + d_ac*ac_count + 
                    d_ca*ca_count + d_bc*bc_count + d_cb*cb_count)
        
        net_transitions[i] = net_count
    
    # Compute mean and variance of net transitions
    mean_net = np.mean(net_transitions)
    
    if mean_net == 0:
        return float('inf')
    
    mean_net_sq = np.mean(net_transitions**2)
    variance = mean_net_sq - mean_net**2
    uncertainty_squared = variance / (mean_net**2)
    
    if uncertainty_squared <= 0:
        return float('inf')
    
    
    return 2 / (uncertainty_squared * snippet_time_length)


# ============================================================================
# Rate Matrix Utilities
# ============================================================================

def generate_rate_matrix_for_parallel_tracks(alpha, beta, u_one, w_one, u_two, w_two):
    """
    Generate rate matrix for parallel tracks model, as described in the paper by Oleg Igoshin and Dima.
    
    Args:
        alpha, beta: Transition rates between tracks
        u_one, w_one: Forward and backward rates on track 1
        u_two, w_two: Forward and backward rates on track 2
        
    Returns:
        6x6 rate matrix for the parallel tracks system
    """
    rate_matrix = np.array([
        [-(u_one+w_one+beta),   alpha,   w_one,   0.0,  u_one,   0.0],
        [  beta,  -(u_two+w_two+alpha),   0.0,   w_two,   0.0,   u_two],
        [ u_one,   0.0, -(u_one+w_one+beta),   alpha,   w_one,   0.0],
        [  0.0,   u_two,   beta,  -(u_two+w_two+alpha),   0.0,   w_two],
        [  w_one,   0.0,  u_one,   0.0, -(u_one+w_one+beta),   alpha],
        [  0.0,   w_two,   0.0,   u_two,   beta,  -(u_two+w_two+alpha)],
    ])
    
    return rate_matrix


def get_true_EPR_for_parallel_tracks(alpha, beta, u_one, w_one, u_two, w_two):
    """
    Calculate exact entropy production rate for parallel tracks model.
    
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


def mathematica_to_numpy_array(mathematica_str):
    """
    Convert Mathematica matrix format to NumPy array.
    
    Useful for copying rate matrices directly from Mathematica output.
    Only handles numerical matrices (not symbolic expressions).
    
    Args:
        mathematica_str: String representation of Mathematica matrix
        
    Returns:
        NumPy array
        
    Example:
        >>> matrix_str = "{{-1.5, 0.5}, {0.5, -0.5}}"
        >>> mathematica_to_numpy_array(matrix_str)
    """
    # Remove outer braces and split into rows
    rows = mathematica_str.strip('{}').split('}, {')
    
    # Parse each row
    data = []
    for row in rows:
        numbers = [float(num) for num in row.replace('{', '').replace('}', '').split(',')]
        data.append(numbers)
    
    return np.array(data)


def print_mathematica_array_as_numpy_form(mathematica_input):
    """
    Print Mathematica array in NumPy-friendly format.
    
    Converts and pretty-prints a Mathematica matrix for easy copy-paste into Python.
    
    Args:
        mathematica_input: String representation of Mathematica matrix
    """
    rate_matrix = mathematica_to_numpy_array(mathematica_input)
    print("rate_matrix = np.array([")
    for row in rate_matrix:
        formatted_row = ', '.join(f"{val:8.10f}" for val in row)
        print(f"    [{formatted_row}],")
    print("])")


# ============================================================================
# Trajectory Processing and Export
# ============================================================================

def coarse_grain_trajectory(trajectory, times, state_mapping=None):
    """
    Coarse-grain trajectory by grouping consecutive identical mapped states.
    
    Args:
        trajectory: List of fine-grained states
        times: List of arrival times
        state_mapping: Dict mapping states to coarse-grained labels
        
    Returns:
        coarse_trajectory: List of coarse-grained states
        coarse_times: List of entry times for each coarse state
        coarse_durations: List of residence times in each coarse state
    """
    # Default mapping for 3-metastate systems
    if state_mapping is None:
        state_mapping = {0: "A", 1: "A", 2: "B", 3: "B", 4: "C", 5: "C"}
    
    # Apply state mapping
    converted_trajectory = [state_mapping.get(state, state) for state in trajectory]
    
    coarse_trajectory = []
    coarse_times = []
    coarse_durations = []
    
    # Group consecutive identical states
    for mapped_state, group in groupby(enumerate(converted_trajectory), key=lambda x: x[1]):
        group_indices = [i for i, _ in group]
        
        coarse_trajectory.append(mapped_state)
        start_time = times[group_indices[0]]
        coarse_times.append(start_time)
        
        # Calculate duration
        if group_indices[-1] + 1 < len(times):
            end_time = times[group_indices[-1] + 1]
            duration = end_time - start_time
        else:
            # Estimate duration for last state
            if len(group_indices) > 1:
                duration = times[group_indices[-1]] - times[group_indices[-2]]
            else:
                duration = 0.0
        
        coarse_durations.append(duration)
    
    return coarse_trajectory, coarse_times, coarse_durations


def save_trajectory_to_text(trajectory, times, filename, state_mapping=None):
    """
    Save trajectory to text file in simple format.
    
    Format: State   ArrivalTime
    
    Args:
        trajectory: List of states
        times: List of arrival times
        filename: Output file path
        state_mapping: Optional dict to map state indices to labels
    """
    # Apply mapping if provided
    if state_mapping is not None:
        mapped_trajectory = [state_mapping.get(state, state) for state in trajectory]
    else:
        mapped_trajectory = trajectory
    
    # Create directory if needed
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write file
    with open(filename, 'w') as f:
        for state, time in zip(mapped_trajectory, times):
            f.write(f"{state}\t{time:.6f}\n")
    
    print(f"Trajectory saved to {filename}")
    print(f"Total transitions: {len(trajectory)}")
    print(f"Time range: {times[0]:.6f} to {times[-1]:.6f}")


def save_coarse_trajectory_to_text(coarse_trajectory, coarse_times, filename, 
                                   include_dwell_times=True):
    """
    Save coarse-grained trajectory to text file.
    
    Args:
        coarse_trajectory: List of coarse states
        coarse_times: List of entry times
        filename: Output file path
        include_dwell_times: Whether to include dwell time column
    """
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(filename, 'w') as f:
        if include_dwell_times:
            f.write("State\tStart_Time\tDuration\n")
            coarse_durations = calculate_coarse_grained_dwell_times(coarse_times)
            for state, time, duration in zip(coarse_trajectory, coarse_times, coarse_durations):
                f.write(f"{state}\t{time:.6f}\t{duration:.6f}\n")
        else:
            for state, time in zip(coarse_trajectory, coarse_times):
                f.write(f"{state}\t{time:.6f}\n")
    
    print(f"Coarse-grained trajectory saved to {filename}")


def calculate_coarse_grained_dwell_times(coarse_times):
    """
    Calculate dwell times from coarse-grained entry times.
    
    Args:
        coarse_times: List of times when each coarse state begins
        
    Returns:
        List of dwell times
    """
    durations = []
    for i in range(len(coarse_times) - 1):
        durations.append(coarse_times[i + 1] - coarse_times[i])
    
    # Estimate duration for last state
    if len(coarse_times) > 1:
        durations.append(durations[-1])
    else:
        durations.append(0.0)
    
    return durations


def process_simulation_results(trajectory, times, state_mapping=None, 
                               output_dir="output", debug_prints=False):
    """
    Complete post-processing pipeline for simulation results.
    
    Creates both true, underlying trajectory and coarse-grained trajectory files.
    
    Args:
        trajectory: List of states from simulation
        times: List of times from simulation
        state_mapping: Optional state mapping dictionary
        output_dir: Directory for output files
        debug_prints: Whether to print processing statistics
        
    Returns:
        Dictionary containing all trajectories and file paths
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save original trajectory
    original_file = os.path.join(output_dir, "original_trajectory.txt")
    save_trajectory_to_text(trajectory, times, original_file, state_mapping)
    
    # Create and save coarse-grained trajectory
    coarse_trajectory, coarse_times, coarse_durations = coarse_grain_trajectory(
        trajectory, times, state_mapping
    )
    
    coarse_file = os.path.join(output_dir, "coarse_trajectory.txt")
    save_coarse_trajectory_to_text(coarse_trajectory, coarse_times, coarse_file)
    
    simple_coarse_file = os.path.join(output_dir, "simple_coarse_trajectory.txt")
    save_coarse_trajectory_to_text(coarse_trajectory, coarse_times, 
                                   simple_coarse_file, include_dwell_times=False)
    
    results = {
        'original_trajectory': trajectory,
        'original_times': times,
        'coarse_trajectory': coarse_trajectory,
        'coarse_times': coarse_times,
        'coarse_durations': coarse_durations,
        'files_created': [original_file, coarse_file, simple_coarse_file]
    }
    
    if debug_prints:
        print("\n=== Processing Summary ===")
        print(f"Original trajectory length: {len(trajectory)}")
        print(f"Coarse trajectory length: {len(coarse_trajectory)}")
        print(f"Compression ratio: {len(trajectory) / len(coarse_trajectory):.2f}x")
        print(f"Files created in '{output_dir}':")
        for file in results['files_created']:
            print(f"  - {os.path.basename(file)}")
    
    return results
