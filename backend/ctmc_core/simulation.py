"""
Continuous-Time Markov Chain simulation core.

Port of the research module's simulation functions. The Numba-JIT functions
below are copied verbatim from `docs/source/reference_ctmc_simulator.py` and
must not be modified. The user-facing `simulate_single_trajectory` wrapper is
refactored into a pure function that raises `ValueError` on invalid input
instead of printing warnings and continuing.
"""

from dataclasses import dataclass

import numpy as np
from numba import jit


# ============================================================================
# Core Simulation Functions (JIT-Compiled) — copied verbatim
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

    Orientation convention: element (i, j) is the rate from state j → state i
    (column = starting state, row = ending state).

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


@jit(nopython=True)
def _seed_random_core(seed):
    """Seed the JIT global RNG. Numba's ``np.random`` inside nopython mode
    keeps its own global state that ``np.random.seed`` in Python does not
    touch; this wrapper is the supported way to seed it."""
    np.random.seed(seed)


def seed_simulation(seed):
    """
    Seed the global RNG used by the JIT simulation core.

    Numba's nopython-mode ``np.random`` is not synchronized with the Python
    ``np.random.seed`` call, so simulations would otherwise be unreproducible.
    This helper makes ``simulate_single_trajectory`` deterministic for a given
    seed (used by tests and by the API's warm-up).
    """
    _seed_random_core(int(seed))


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


# ============================================================================
# Validation
# ============================================================================

def validate_rate_matrix(rate_matrix):
    """
    Validate a rate matrix and return it as a float64 numpy array.

    Orientation convention: element (i, j) is the rate from state j → state i
    (column = starting state, row = ending state), so each column sums to zero.

    Raises ``ValueError`` (not warnings) on any malformed or unbalanced input
    so the API can surface a clear HTTP 422. Checks, in order:

    - square, at least 2 states, all entries finite
    - off-diagonal entries non-negative
    - diagonal entries strictly negative (positive, finite exit rates)
    - each column sums to zero
    """
    matrix = np.array(rate_matrix, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Rate matrix must be square")
    if matrix.shape[0] < 2:
        raise ValueError("Rate matrix must have at least 2 states")

    n_states = matrix.shape[0]

    if not np.isfinite(matrix).all():
        for i in range(n_states):
            for j in range(n_states):
                if not np.isfinite(matrix[i, j]):
                    raise ValueError(
                        f"Rate matrix entry ({j}, {i}) must be finite, got {matrix[i, j]}"
                    )

    for i in range(n_states):
        for j in range(n_states):
            if i != j and matrix[i, j] < 0:
                raise ValueError(
                    f"Off-diagonal rate matrix entry ({i}, {j}) must be non-negative, "
                    f"got {matrix[i, j]}"
                )

    exit_rates = -np.diag(matrix)
    for i, exit_rate in enumerate(exit_rates):
        if not np.isfinite(exit_rate):
            raise ValueError(f"Exit rate for state {i} must be finite, got {exit_rate}")
        if exit_rate <= 0:
            raise ValueError(
                f"Exit rate for state {i} must be strictly positive, got {exit_rate}"
            )

    column_sums = matrix.sum(axis=0)
    for i, column_sum in enumerate(column_sums):
        if not np.isclose(column_sum, 0.0, rtol=1e-10, atol=1e-10):
            raise ValueError(
                f"Rate matrix column {i} must sum to zero, got {column_sum}"
            )

    return matrix


def validate_metastate_groups(metastate_groups, n_states):
    """
    Validate that metastate groups partition every state exactly once.

    Returns ``(metastate_names, state_to_metastate_array)`` where
    ``state_to_metastate_array[i]`` is the index (in group definition order) of
    the metastate containing state ``i``.

    Raises ``ValueError`` on out-of-range, duplicate, or missing state
    assignments.
    """
    if not isinstance(metastate_groups, dict) or len(metastate_groups) == 0:
        raise ValueError("Metastate groups must be a non-empty dict")

    metastate_names = list(metastate_groups.keys())
    state_to_metastate = {}

    for metastate_idx, (metastate_name, state_list) in enumerate(metastate_groups.items()):
        for state_idx in state_list:
            if state_idx >= n_states or state_idx < 0:
                raise ValueError(
                    f"State index {state_idx} in metastate {metastate_name!r} "
                    f"exceeds matrix size ({n_states} states)"
                )
            if state_idx in state_to_metastate:
                raise ValueError(
                    f"State {state_idx} is assigned to multiple metastates "
                    f"({state_to_metastate[state_idx]} and {metastate_idx})"
                )
            state_to_metastate[state_idx] = metastate_idx

    missing = [i for i in range(n_states) if i not in state_to_metastate]
    if missing:
        raise ValueError(
            f"State(s) {missing} are not assigned to any metastate group"
        )

    state_to_metastate_array = np.array(
        [state_to_metastate[i] for i in range(n_states)], dtype=np.int32
    )
    return metastate_names, state_to_metastate_array


# ============================================================================
# Result type
# ============================================================================

@dataclass(frozen=True)
class TrajectoryResult:
    """Structured result of a single trajectory simulation."""

    trajectory: np.ndarray
    times: np.ndarray
    waiting_times_by_metastate: dict
    time_in_metastates: dict
    metastate_transition_count: int
    final_time: float


def process_metastate_waiting_times(metastate_changes, waiting_times_raw, metastate_names):
    """
    Aggregate metastate residence times into per-metastate statistics.

    Args:
        metastate_changes: Array of metastate indices visited
        waiting_times_raw: Array of residence times
        metastate_names: Ordered metastate names (index i names metastate i)

    Returns:
        waiting_times_by_metastate: Lists of residence times per metastate
        time_in_metastates: Total time spent in each metastate
        metastate_transition_count: Number of metastate transitions
    """
    waiting_times_by_metastate = {name: [] for name in metastate_names}

    for i in range(len(metastate_changes)):
        metastate_idx = metastate_changes[i]
        waiting_time = waiting_times_raw[i]

        if metastate_idx < len(metastate_names) and waiting_time > 0:
            metastate_name = metastate_names[metastate_idx]
            waiting_times_by_metastate[metastate_name].append(waiting_time)

    time_in_metastates = {
        metastate: float(np.sum(waiting_list))
        for metastate, waiting_list in waiting_times_by_metastate.items()
    }

    return waiting_times_by_metastate, time_in_metastates, len(metastate_changes)


# ============================================================================
# Main Simulation Interface
# ============================================================================

def simulate_single_trajectory(rate_matrix, metastate_groups, max_length=1000,
                              initial_state=0):
    """
    Simulate a continuous-time Markov chain trajectory with metastate tracking.

    Pure function: no printing, no timing, no file I/O. Invalid input raises
    ``ValueError``.

    Args:
        rate_matrix: Square matrix where element (i,j) is the rate from j to i
            (column = starting state, row = ending state)
        metastate_groups: Dict mapping metastate names to lists of state indices
        max_length: Maximum trajectory length (number of transitions)
        initial_state: Starting state index

    Returns:
        TrajectoryResult with trajectory, times, per-metastate residence
        statistics, and final time.
    """
    if not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")

    matrix = validate_rate_matrix(rate_matrix)
    n_states = matrix.shape[0]

    if not isinstance(initial_state, int) or not (0 <= initial_state < n_states):
        raise ValueError(
            f"initial_state must be an integer in [0, {n_states - 1}], got {initial_state}"
        )

    metastate_names, state_to_metastate_array = validate_metastate_groups(
        metastate_groups, n_states
    )

    exit_rates, transition_matrices = precompute_transition_data(matrix)

    trajectory, times, metastate_changes, waiting_times_raw = simulate_trajectory_core(
        matrix, state_to_metastate_array, max_length, initial_state,
        exit_rates, transition_matrices
    )

    waiting_times_by_metastate, time_in_metastates, metastate_transition_count = (
        process_metastate_waiting_times(
            metastate_changes, waiting_times_raw, metastate_names
        )
    )

    final_time = float(times[-1]) if len(times) > 0 else 0.0

    return TrajectoryResult(
        trajectory=trajectory,
        times=times,
        waiting_times_by_metastate=waiting_times_by_metastate,
        time_in_metastates=time_in_metastates,
        metastate_transition_count=metastate_transition_count,
        final_time=final_time,
    )