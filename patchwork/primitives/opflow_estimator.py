"""This is the land of abominations: An opflow based primitive.

But if life gives you shitty lemons, sometimes you just have to grow a new tree I guess.
Or a give an old, good tree, that no one loves a nice wrapping until everything burns down in a
big fire; the wrapping, the old tree, memories and dreams, and then, hopefully, something
better emerges, like a phoenix from the ashes. Or maybe not. But that's life!
"""

from __future__ import annotations
from collections.abc import Iterable, Sequence
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers import Backend
from qiskit.providers import JobV1 as Job
from qiskit.opflow import CircuitSampler, StateFn, ExpectationBase, PauliSumOp, PrimitiveOp, ListOp
from qiskit.utils import QuantumInstance

from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.primitives import Estimator
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives.utils import _circuit_key, _observable_key, init_observable


class OpflowEstimator(BaseEstimator):
    """An estimator based on opflow.

    Why? Hopefully lot's of speed.
    """

    def __init__(
        self,
        circuits: Iterable[QuantumCircuit] | QuantumCircuit | None = None,
        observables: Iterable[SparsePauliOp] | SparsePauliOp | None = None,
        parameters: Iterable[Iterable[Parameter]] | None = None,
        options: dict | None = None,
        expectation_converter: ExpectationBase | None = None,
        backend: QuantumInstance | Backend | None = None,
    ) -> None:
        """
        Args:
            circuits: Not supported.
            observables: Not supported.
            parameters: Not supported.
            options: Not supported.
            expectation_converter: An expectation converter.
            backend: The backend or quantum instance to execute circuits.
        """
        for name, arg in {
            ("circuits", circuits),
            ("observables", observables),
            ("parameters", parameters),
            ("options", options),
        }:
            if arg is not None:
                raise ValueError(f"{name} is not supported in the Opflow primitive.")

        super().__init__(circuits, observables, parameters, options)

        if expectation_converter is None:
            raise ValueError("expectation_converter is required.")

        self.expectation_converter = expectation_converter

        if backend is None:
            raise ValueError("backend is required.")

        if not isinstance(backend, QuantumInstance):
            backend = QuantumInstance(backend)

        self.sampler = CircuitSampler(backend)

        self._expectation_cache = {}

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[BaseOperator | PauliSumOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> Job:
        expectations = {}

        for i, (circuit, observable) in enumerate(zip(circuits, observables)):
            # try fetching the already converted expectation
            observable = init_observable(observable)
            key = (_circuit_key(circuit), _observable_key(observable))
            exp = self._expectation_cache.get(key, None)

            # it if did not exist, build it
            if exp is None:
                if not isinstance(observable, PauliSumOp):
                    observable = PrimitiveOp(observable)
                exp = StateFn(observable, is_measurement=True).compose(StateFn(circuit))
                exp = self.expectation_converter.convert(exp)
                self._expectation_cache[key] = exp

            # generate a dictionary with the expectation as key,
            # and as value a tuple of (parameters, [index1, index2, ...], [values1, values2, ...])
            if key not in expectations.keys():
                expectations[key] = (exp, circuit.parameters, [i], [parameter_values[i]])
            else:
                expectations[key][2].append(i)
                expectations[key][3].append(parameter_values[i])

        job = PrimitiveJob(self._evaluate, expectations, len(circuits))
        job.submit()

        return job

    def _evaluate(self, grouped, num_results) -> EstimatorResult:
        values = [None] * num_results
        metadata = [None] * num_results
        for _, (expectation, parameters, indices, values_list) in grouped.items():
            transposed = np.array(values_list).T.tolist()
            param_dict = dict(zip(parameters, transposed))
            sampled = self.sampler.convert(expectation, params=param_dict)
            std = self.expectation_converter.compute_variance(sampled)
            value = sampled.eval()

            for i, value_i, std_i in zip(indices, value, std):
                values[i] = value_i
                metadata[i] = {"std": std_i}

        result = EstimatorResult(values, metadata=metadata)
        return result

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        raise RuntimeError("I don't need this mate!")
