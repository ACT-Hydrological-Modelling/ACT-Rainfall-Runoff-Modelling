"""
Unified parallel backend for pyrrm experiment runners.

Provides a pluggable parallelization abstraction used by both
BatchExperimentRunner and CatchmentNetworkRunner. Three backends:

- SequentialBackend: No parallelism (default, zero dependencies)
- MultiprocessingBackend: stdlib ProcessPoolExecutor
- RayBackend: Ray task-graph-aware scheduling (optional dependency)

Example:
    >>> from pyrrm.parallel import create_backend
    >>> backend = create_backend('sequential')
    >>> results = backend.map(my_func, tasks)
    >>> backend.shutdown()
"""

from abc import ABC, abstractmethod
from typing import (
    TypeVar, Generic, Callable, List, Dict, Optional, Any, Tuple,
)
import logging
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class ParallelBackend(ABC):
    """Abstract interface for parallel task execution."""

    @abstractmethod
    def map(
        self,
        func: Callable,
        tasks: List,
        task_ids: Optional[List[str]] = None,
        on_complete: Optional[Callable[[str, Any], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> Dict[str, Any]:
        """Execute func on each task, return results keyed by task_id.

        Args:
            func: Callable that takes a single task and returns a result.
            tasks: List of task arguments.
            task_ids: Optional list of string IDs (one per task).
                If None, uses str(index).
            on_complete: Callback(task_id, result) after each success.
            on_error: Callback(task_id, exception) after each failure.

        Returns:
            Dict mapping task_id to result for successful tasks.
        """

    @abstractmethod
    def map_wavefronts(
        self,
        func: Callable,
        wavefronts: List[List[Tuple[str, Any]]],
        on_complete: Optional[Callable[[str, Any], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> Dict[str, Any]:
        """Execute tasks in dependency-ordered wavefronts.

        Each wavefront is a list of (task_id, task_arg) tuples.
        All tasks in a wavefront are independent and can run in parallel.
        Wavefronts are processed sequentially: wavefront N must complete
        before wavefront N+1 starts.

        Args:
            func: Callable that takes a single task arg and returns a result.
            wavefronts: List of wavefronts, each a list of (id, arg) tuples.
            on_complete: Callback(task_id, result) after each success.
            on_error: Callback(task_id, exception) after each failure.

        Returns:
            Dict mapping task_id to result for all successful tasks.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""


class SequentialBackend(ParallelBackend):
    """No parallelism. For debugging and small experiments."""

    def map(
        self,
        func: Callable,
        tasks: List,
        task_ids: Optional[List[str]] = None,
        on_complete: Optional[Callable[[str, Any], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> Dict[str, Any]:
        if task_ids is None:
            task_ids = [str(i) for i in range(len(tasks))]

        results: Dict[str, Any] = {}
        for tid, task in zip(task_ids, tasks):
            try:
                result = func(task)
                results[tid] = result
                if on_complete:
                    on_complete(tid, result)
            except Exception as e:
                logger.error("Task %s failed: %s", tid, e)
                if on_error:
                    on_error(tid, e)
                else:
                    raise
        return results

    def map_wavefronts(
        self,
        func: Callable,
        wavefronts: List[List[Tuple[str, Any]]],
        on_complete: Optional[Callable[[str, Any], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> Dict[str, Any]:
        all_results: Dict[str, Any] = {}
        for wf_idx, wavefront in enumerate(wavefronts):
            ids = [tid for tid, _ in wavefront]
            args = [arg for _, arg in wavefront]
            wf_results = self.map(func, args, task_ids=ids,
                                  on_complete=on_complete, on_error=on_error)
            all_results.update(wf_results)
            logger.info("Wavefront %d/%d complete (%d tasks)",
                        wf_idx + 1, len(wavefronts), len(wavefront))
        return all_results

    def shutdown(self) -> None:
        pass


class MultiprocessingBackend(ParallelBackend):
    """Parallel execution via concurrent.futures.ProcessPoolExecutor."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def map(
        self,
        func: Callable,
        tasks: List,
        task_ids: Optional[List[str]] = None,
        on_complete: Optional[Callable[[str, Any], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> Dict[str, Any]:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if task_ids is None:
            task_ids = [str(i) for i in range(len(tasks))]

        results: Dict[str, Any] = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(func, task): tid
                for tid, task in zip(task_ids, tasks)
            }
            for future in as_completed(future_to_id):
                tid = future_to_id[future]
                try:
                    result = future.result()
                    results[tid] = result
                    if on_complete:
                        on_complete(tid, result)
                except Exception as e:
                    logger.error("Task %s failed: %s", tid, e)
                    if on_error:
                        on_error(tid, e)
                    else:
                        raise
        return results

    def map_wavefronts(
        self,
        func: Callable,
        wavefronts: List[List[Tuple[str, Any]]],
        on_complete: Optional[Callable[[str, Any], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> Dict[str, Any]:
        all_results: Dict[str, Any] = {}
        for wf_idx, wavefront in enumerate(wavefronts):
            ids = [tid for tid, _ in wavefront]
            args = [arg for _, arg in wavefront]
            wf_results = self.map(func, args, task_ids=ids,
                                  on_complete=on_complete, on_error=on_error)
            all_results.update(wf_results)
            logger.info("Wavefront %d/%d complete (%d tasks)",
                        wf_idx + 1, len(wavefronts), len(wavefront))
        return all_results

    def shutdown(self) -> None:
        pass


class RayBackend(ParallelBackend):
    """Ray-based parallel execution with DAG-aware scheduling.

    Ray is used for efficient task graph execution on a single machine.
    It handles resource management (CPU allocation per task) and
    dynamic scheduling.

    Requires: ``pip install ray``
    """

    def __init__(self, num_cpus: Optional[int] = None, **ray_init_kwargs):
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is required for RayBackend but is not installed. "
                "Install with: pip install ray"
            )
        if not ray.is_initialized():
            init_kwargs = {}
            if num_cpus is not None:
                init_kwargs['num_cpus'] = num_cpus
            init_kwargs.update(ray_init_kwargs)
            ray.init(**init_kwargs)
            self._initialized_ray = True
        else:
            self._initialized_ray = False
        self.num_cpus = num_cpus

    def map(
        self,
        func: Callable,
        tasks: List,
        task_ids: Optional[List[str]] = None,
        on_complete: Optional[Callable[[str, Any], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> Dict[str, Any]:
        if task_ids is None:
            task_ids = [str(i) for i in range(len(tasks))]

        remote_func = ray.remote(func)

        futures = {}
        for tid, task in zip(task_ids, tasks):
            futures[remote_func.remote(task)] = tid

        results: Dict[str, Any] = {}
        ready_refs = list(futures.keys())

        while ready_refs:
            done, ready_refs = ray.wait(ready_refs, num_returns=1)
            for ref in done:
                tid = futures[ref]
                try:
                    result = ray.get(ref)
                    results[tid] = result
                    if on_complete:
                        on_complete(tid, result)
                except Exception as e:
                    logger.error("Task %s failed: %s", tid, e)
                    if on_error:
                        on_error(tid, e)
                    else:
                        raise
        return results

    def map_wavefronts(
        self,
        func: Callable,
        wavefronts: List[List[Tuple[str, Any]]],
        on_complete: Optional[Callable[[str, Any], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> Dict[str, Any]:
        all_results: Dict[str, Any] = {}
        for wf_idx, wavefront in enumerate(wavefronts):
            ids = [tid for tid, _ in wavefront]
            args = [arg for _, arg in wavefront]
            wf_results = self.map(func, args, task_ids=ids,
                                  on_complete=on_complete, on_error=on_error)
            all_results.update(wf_results)
            logger.info("Wavefront %d/%d complete (%d tasks)",
                        wf_idx + 1, len(wavefronts), len(wavefront))
        return all_results

    def shutdown(self) -> None:
        if self._initialized_ray and ray.is_initialized():
            ray.shutdown()


def create_backend(
    backend: str = 'sequential',
    max_workers: Optional[int] = None,
    **kwargs,
) -> ParallelBackend:
    """Create a parallel backend by name.

    Args:
        backend: One of 'sequential', 'multiprocessing', 'ray'.
        max_workers: Number of parallel workers (for multiprocessing/ray).
        **kwargs: Additional keyword arguments passed to the backend constructor.

    Returns:
        A ParallelBackend instance.

    Raises:
        ValueError: If backend name is not recognized.
        ImportError: If Ray backend is requested but Ray is not installed.

    Example:
        >>> backend = create_backend('multiprocessing', max_workers=4)
        >>> results = backend.map(my_func, tasks)
        >>> backend.shutdown()
    """
    backend_lower = backend.lower()
    if backend_lower == 'sequential':
        return SequentialBackend()
    elif backend_lower == 'multiprocessing':
        return MultiprocessingBackend(max_workers=max_workers or 4)
    elif backend_lower == 'ray':
        return RayBackend(num_cpus=max_workers, **kwargs)
    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            f"Choose from: 'sequential', 'multiprocessing', 'ray'"
        )


__all__ = [
    'ParallelBackend',
    'SequentialBackend',
    'MultiprocessingBackend',
    'RayBackend',
    'RAY_AVAILABLE',
    'create_backend',
]
