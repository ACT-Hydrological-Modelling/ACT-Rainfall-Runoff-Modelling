"""An implementation of SCE-UA Algorithm for single-objective optimization.

Vendored from https://github.com/cheginit/sceua with minor modifications
for compatibility with pyrrm's numpy/scipy version requirements.

References
----------
- Duan, Q., Sorooshian, S., & Gupta, V. K. (1992). Effective and efficient global
    optimization for conceptual rainfall-runoff models. Water Resources Research, 28(4),
    1015-1031. doi:10.1029/91WR02985
- Duan, Q., Gupta, V. K., & Sorooshian, S. (1994). Optimal use of the SCE-UA global
    optimization method for calibrating watershed models. Journal of Hydrology,
    158(3-4), 265-284. doi:10.1016/0022-1694(94)90057-4
- Duan, Q., Sorooshian, S., & Gupta, V. K. (1994). A shuffled complex evolution approach
    for effective and efficient global minimization. Journal of optimization theory and
    applications, 76(3), 501-521. doi:10.1007/BF00939380
- Muttil, N., & Jayawardena, A. W. (2008). Shuffled Complex Evolution model calibrating
    algorithm: enhancing its robustness and efficiency. Hydrological Processes, 22(23),
    4628-4638. doi:10.1002/hyp.7082
- Chu, W., Gao, X., & Sorooshian, S. (2010). Improving the shuffled complex evolution
    scheme for optimization of complex nonlinear hydrological systems: Application to
    the calibration of the Sacramento soil-moisture accounting model. Water Resources
    Research, 46(9). doi:10.1029/2010wr009224
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import qmc

# Try to import tqdm for progress bars
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating[Any]]
    IntArray = NDArray[np.integer[Any]]
    FuncType = Union[Callable[[FloatArray, Any], float], Callable[[FloatArray], float]]
    CallbackType = Callable[[dict], None]


@dataclass
class Result:
    """Result object for the optimization.

    Attributes
    ----------
    x : numpy.ndarray
        Best parameters found.
    fun : float
        Best function value corresponding to the best parameters.
    nit : int
        Number of iterations.
    nfev : int
        Number of function evaluations.
    message : str
        Message describing the termination reason.
    success : bool
        Whether the optimization was successful.
    xv : numpy.ndarray
        All evaluated parameter sets.
    funv : numpy.ndarray
        Function values for all evaluated parameter sets.
    """

    x: "FloatArray"
    fun: float
    nit: int
    nfev: int
    message: str
    success: bool
    xv: "FloatArray"
    funv: "FloatArray"


def _generate_population(
    lo_bounds: "FloatArray",
    up_bounds: "FloatArray",
    n: int,
    x0: Optional[Union["FloatArray", List[Tuple[float, ...]]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> "FloatArray":
    """Initialize population using Latin Hypercube Sampling."""
    n_dim = len(lo_bounds)

    if x0 is None:
        x0_arr = np.empty((0, n_dim))
    else:
        x0_arr = np.atleast_2d(x0)
        if x0_arr.shape[1] != n_dim:
            raise ValueError(f"x0 must have shape (n, {n_dim})")

    remaining = max(0, n - len(x0_arr))
    if remaining == 0:
        return x0_arr[:n]

    if rng is None:
        rng = np.random.default_rng()

    x_samples = qmc.LatinHypercube(n_dim, rng=rng).random(remaining)
    x_samples = qmc.scale(x_samples, lo_bounds, up_bounds)
    return np.vstack([x0_arr, x_samples]) if len(x0_arr) > 0 else x_samples


def _evolve_complexes(
    func: "FuncType",
    args: Tuple[Any, ...],
    population: "FloatArray",
    func_values: "FloatArray",
    complex_indices: "IntArray",
    lo_bounds: "FloatArray",
    up_bounds: "FloatArray",
    rng: np.random.Generator,
    alpha: float,
    beta: float,
) -> None:
    """Competitive Complex Evolution (CCE) for a given complex.

    Includes improvements:
    1. Adaptive smoothing parameter (theta) based on problem scale
    2. Best solution is included in every complex
    """
    # Determine theta following the suggestion
    # from https://github.com/sambo-optimization/sambo
    # which is based on https://onlinelibrary.wiley.com/doi/10.1002/hyp.7082
    theta = np.interp(
        np.log10(np.ptp(np.c_[lo_bounds, up_bounds], axis=1).max()), (2, 5), (0.2, 0.5)
    )

    # Extract complex from population
    complex_population = population[complex_indices].copy()
    complex_values = func_values[complex_indices].copy()

    # Best and worst indices
    best_idx, worst_idx = 0, -1

    # Calculate centroid excluding worst point
    centroid = np.mean(complex_population[:worst_idx], axis=0)
    reflection = centroid + alpha * (centroid - complex_population[worst_idx])
    # Apply smoothing with best solution (Beven, 2006 - doi:10.1002/hyp.7082)
    reflection = (1 - theta) * reflection + theta * complex_population[best_idx]
    reflection = np.clip(reflection, lo_bounds, up_bounds)
    reflection_value = func(reflection, *args)

    # If reflection is better than worst, replace worst point
    if reflection_value < complex_values[worst_idx]:
        complex_population[worst_idx] = reflection
        complex_values[worst_idx] = reflection_value
    else:
        contraction = complex_population[worst_idx] + beta * (
            centroid - complex_population[worst_idx]
        )
        # Apply smoothing with best solution
        contraction = (1 - theta) * contraction + theta * complex_population[best_idx]
        contraction = np.clip(contraction, lo_bounds, up_bounds)
        contraction_value = func(contraction, *args)

        # If contraction is better than worst, replace worst point
        if contraction_value < complex_values[worst_idx]:
            complex_population[worst_idx] = contraction
            complex_values[worst_idx] = contraction_value
        else:
            # Random point if both reflection and contraction fail
            n_dim = len(lo_bounds)
            random_point = lo_bounds + rng.random(n_dim) * (up_bounds - lo_bounds)
            random_value = func(random_point, *args)
            complex_population[worst_idx] = random_point
            complex_values[worst_idx] = random_value

    sorted_indices = np.argsort(complex_values)
    complex_population = complex_population[sorted_indices]
    complex_values = complex_values[sorted_indices]

    population[complex_indices] = complex_population
    func_values[complex_indices] = complex_values


def _pca_recovery(
    population: "FloatArray",
    lo_bounds: "FloatArray",
    up_bounds: "FloatArray",
    tol: float,
    rng: np.random.Generator,
) -> None:
    """Recover lost dimensions in the population.

    Based on https://doi.org/10.1029/2010WR009224.

    Parameters
    ----------
    population : numpy.ndarray
        Current population of parameter vectors.
    lo_bounds, up_bounds : numpy.ndarray
        Lower and upper bounds.
    tol : float
        Tolerance factor to decide if a dimension is lost, by comparing
        each eigenvalue to ``tol * max(eigenvalue)``.
    rng : numpy.random.Generator
        Random number generator.
    """
    if population.shape[0] < 2:
        return

    mean_vec = np.mean(population, axis=0)
    x_centered = population - mean_vec
    cov = np.cov(x_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    max_eig = np.max(eigenvalues)

    # Check for any lost (nearly zero variance) dimensions.
    for i, eig in enumerate(eigenvalues):
        if eig < tol * max_eig:
            direction = eigenvectors[:, i]
            n_indiv = population.shape[0]
            n_to_perturb = max(1, int(0.1 * n_indiv))  # perturb 10% of individuals
            indices = rng.choice(n_indiv, n_to_perturb, replace=False)
            factor = 0.05 * (up_bounds - lo_bounds).mean()  # scale factor
            for idx in indices:
                scalar = rng.random() * factor
                population[idx] = np.clip(
                    population[idx] + scalar * direction, lo_bounds, up_bounds
                )
            break


def _update_population(
    pop_x: "FloatArray",
    pop_f: "FloatArray",
    iters: int,
    pca_freq: int,
    pca_tol: float,
    rng: np.random.Generator,
    lo_bounds: "FloatArray",
    up_bounds: "FloatArray",
    best_f: float,
    best_x: "FloatArray",
    last_best_f: float,
    no_change_count: int,
    tolerance: float,
) -> Tuple["FloatArray", "FloatArray", "FloatArray", float, int]:
    """Update population based on the best solution found so far and PCA recovery."""
    idx = np.argsort(pop_f)
    pop_x, pop_f = pop_x[idx], pop_f[idx]

    # Check effective dimensionality from https://doi.org/10.1029/2010WR009224
    if iters % pca_freq == 0:
        cov = np.cov(pop_x, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        ratios = eigenvalues / eigenvalues.max()
        lost_dims = np.sum(ratios < pca_tol)
        if lost_dims > 0:
            _pca_recovery(pop_x, lo_bounds, up_bounds, pca_tol, rng)

    if pop_f[0] < best_f:
        best_x = pop_x[0].copy()
        best_f = pop_f[0]
        no_change_count = 0 if abs(last_best_f - best_f) > tolerance else no_change_count + 1
    else:
        no_change_count += 1

    return pop_x, pop_f, best_x, best_f, no_change_count


def minimize(
    func: "FuncType",
    bounds: Sequence[Tuple[float, float]],
    *,
    args: Tuple[Any, ...] = (),
    n_complexes: Optional[int] = None,
    n_points_complex: Optional[int] = None,
    alpha: float = 1.0,
    beta: float = 0.5,
    max_evals: int = 50000,
    max_iter: int = 1000,
    max_tolerant_iter: int = 30,
    tolerance: float = 1e-6,
    x_tolerance: float = 1e-8,
    seed: Optional[int] = None,
    pca_freq: int = 1,
    pca_tol: float = 1e-3,
    x0: Optional[Union["FloatArray", List[Tuple[float, ...]]]] = None,
    max_workers: int = 1,
    callback: Optional["CallbackType"] = None,
    verbose: bool = False,
    progress_bar: bool = True,
    display_maximize: bool = False,
) -> Result:
    """Minimize a function using an improved SCE-UA algorithm.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.
        ``func(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and args are positional arguments.
    bounds : sequence of tuples
        Bounds for variables, (min, max) pairs for each element in ``x``.
    args : tuple, optional
        Extra arguments passed to the objective function, defaults to ``()``.
    n_complexes : int, optional
        Number of complexes, defaults to ``None`` (adaptive based on dimensionality).
    n_points_complex : int, optional
        Number of points in each complex. If ``None``, calculated automatically.
    alpha : float, optional
        Reflection coefficient, defaults to 1.0.
    beta : float, optional
        Contraction coefficient, defaults to 0.5.
    max_evals : int, optional
        Maximum number of function evaluations, defaults to 50000.
    max_iter : int, optional
        Maximum number of iterations, defaults to 1000.
    max_tolerant_iter : int, optional
        Maximum number of tolerated iterations without improvement, defaults to 30.
    tolerance : float, optional
        Tolerance for improvement in function value, defaults to 1e-6.
    x_tolerance : float, optional
        Tolerance for parameter convergence (parameter space), defaults to 1e-8.
    seed : int, optional
        Random seed for reproducibility, defaults to ``None``.
    pca_freq : int, optional
        Frequency of PCA recovery, defaults to 1, i.e., every iterations.
    pca_tol : float, optional
        Tolerance for PCA recovery, defaults to 1e-3.
        This is the threshold for eigenvalues to be considered significant.
        This threshold is used to apply PCA recovery based on the ratio of
        eigenvalues to the maximum eigenvalue, i.e., ``eig / max(eig) < pca_tol``.
    x0 : numpy.ndarray, optional
        Initial parameter sets to evaluate, defaults to ``None``.
    max_workers : int, optional
        Number of workers for parallel execution, defaults to 1 (no parallelism).
        This is passed to the ``ThreadPoolExecutor``.
    callback : callable, optional
        Function called after each iteration with a dict containing:
        ``{'iteration': int, 'nfev': int, 'best_fun': float, 'best_x': array}``.
    verbose : bool, optional
        If True, print progress information every 10 iterations. Defaults to False.
    progress_bar : bool, optional
        If True and tqdm is available, show a progress bar. Defaults to True.
    display_maximize : bool, optional
        If True, verbose output shows negated values (for maximization objectives
        like NSE where the function is negated for minimization). Defaults to False.

    Returns
    -------
    result : Result
        The optimization result with the following fields:

        - ``x`` : numpy.ndarray
            The optimal parameters found.
        - ``fun`` : float
            Best function value corresponding to the best parameters.
        - ``nit`` : int
            Number of iterations.
        - ``nfev`` : int
            Number of function evaluations.
        - ``message`` : str
            Description of the termination reason.
        - ``success`` : bool
            Whether the optimizer exited successfully.
        - ``xv`` : numpy.ndarray
            All parameter vectors that were evaluated.
        - ``funv`` : numpy.ndarray
            Function values for all evaluated parameter vectors.

    Examples
    --------
    >>> import numpy as np
    >>> from pyrrm.calibration._sceua import minimize
    >>> 
    >>> # Define a simple objective function (Rosenbrock)
    >>> def rosenbrock(x):
    ...     return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    >>> 
    >>> bounds = [(-5, 5), (-5, 5)]
    >>> result = minimize(rosenbrock, bounds, seed=42, max_evals=5000)
    >>> print(f"Best parameters: {result.x}")
    >>> print(f"Best value: {result.fun}")
    """
    if not callable(func):
        raise TypeError("`func` must be a callable function.")

    try:
        lo_bounds, up_bounds = np.atleast_2d(bounds).T
    except ValueError as e:
        raise ValueError("Bounds must be a sequence of (min, max) pairs.") from e
    if np.any(lo_bounds >= up_bounds):
        raise ValueError("Lower bounds must be less than upper bounds.")

    xv: List["FloatArray"] = []
    funv: List[float] = []

    def objective(x: "FloatArray", *args_inner: Any) -> float:
        """Objective function wrapper to store evaluated points."""
        nonlocal xv, funv
        y = func(x, *args_inner)
        xv.append(x.copy())
        funv.append(y)
        return y

    # Initialize SCE-UA parameters based on suggestions from the literature
    if (isinstance(n_complexes, Number) and n_complexes < 2) or n_complexes is None:
        n_complexes = min(max(2, int(np.log2(len(bounds))) + 5), 15)
    n_points_complex = 2 * n_complexes + 1 if n_points_complex is None else n_points_complex

    rng = np.random.default_rng(seed)
    n_pop = n_points_complex * n_complexes
    pop_x = _generate_population(lo_bounds, up_bounds, n_pop, x0, rng)
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pop_f = np.asarray(list(executor.map(lambda x: objective(x, *args), pop_x)))
    else:
        pop_f = np.asarray([objective(x, *args) for x in pop_x])

    idx = np.argsort(pop_f)
    pop_x, pop_f = pop_x[idx], pop_f[idx]
    best_x, best_f = pop_x[0].copy(), pop_f[0]
    last_best_f = best_f

    iters = 0
    no_change_count = 0
    message = "Maximum iterations reached"
    n_evals = len(xv)
    max_evals = max(max_evals, n_pop * 2)
    
    # Setup progress bar if available and requested
    pbar = None
    if progress_bar and TQDM_AVAILABLE:
        pbar = tqdm(
            total=max_evals,
            initial=n_evals,
            desc="SCE-UA",
            unit="eval",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} evals [{elapsed}<{remaining}, best={postfix}]"
        )
        pbar.set_postfix_str(f"{best_f:.4f}")
    
    # display_maximize is used for verbose output when the objective should be maximized (e.g., NSE)
    # When True, the displayed values are negated to show actual objective values
    display_max = display_maximize
    
    if verbose:
        display_f = -best_f if display_max else best_f
        display_mode = "maximizing" if display_max else "minimizing"
        _log.info("SCE-UA started | params=%d complexes=%d pop=%d max_evals=%d",
                  len(lo_bounds), n_complexes, n_pop, max_evals)
        _log.info("Initial best (%s): %.6f", display_mode, display_f)
    
    try:
        while n_evals < max_evals and iters < max_iter and no_change_count < max_tolerant_iter:
            # Create complex indices with best solution (index 0) in every complex
            # based on a suggestion from https://github.com/sambo-optimization/sambo
            complex_idx = (np.hstack((0, np.arange(k, n_pop, n_complexes))) for k in range(n_complexes))
            _ = [
                _evolve_complexes(
                    objective, args, pop_x, pop_f, idx, lo_bounds, up_bounds, rng, alpha, beta
                )
                for idx in complex_idx
            ]
            pop_x, pop_f, best_x, best_f, no_change_count = _update_population(
                pop_x,
                pop_f,
                iters,
                pca_freq,
                pca_tol,
                rng,
                lo_bounds,
                up_bounds,
                best_f,
                best_x,
                last_best_f,
                no_change_count,
                tolerance,
            )

            last_best_f = best_f
            iters += 1
            n_evals = len(xv)
            
            # Update progress bar
            if pbar is not None:
                pbar.n = n_evals
                pbar.set_postfix_str(f"{best_f:.4f}")
                pbar.refresh()
            
            if verbose and iters % 10 == 0:
                display_best = -best_f if display_max else best_f
                _log.info("Iter %4d: nfev=%6d, best=%.6f, no_improve=%d",
                          iters, n_evals, display_best, no_change_count)
            
            # Call user callback if provided
            if callback is not None:
                callback({
                    'iteration': iters,
                    'nfev': n_evals,
                    'best_fun': best_f,
                    'best_x': best_x.copy(),
                })

            param_range = np.max(pop_x, axis=0) - np.min(pop_x, axis=0)
            if np.sum(param_range) < x_tolerance:
                message = f"Parameter convergence (sum of ranges < {x_tolerance})"
                break
    finally:
        # Close progress bar
        if pbar is not None:
            pbar.close()

    if n_evals >= max_evals:
        message = f"Maximum function evaluations reached ({max_evals})"
    elif no_change_count >= max_tolerant_iter:
        message = f"No improvement for {max_tolerant_iter} iterations (< {tolerance})"
    
    if verbose:
        _log.info("SCE-UA complete: %s | iters=%d nfev=%d best=%.6f",
                  message, iters, n_evals, best_f)

    return Result(
        x=best_x,
        fun=best_f,
        nit=iters,
        nfev=n_evals,
        message=message,
        success=True,
        xv=np.asarray(xv),
        funv=np.asarray(funv),
    )
