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

from qiskit.primitives import BaseSampler, SamplerResult
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives.utils import _circuit_key, _observable_key, init_observable
from qiskit.result import QuasiDistribution


class OpflowSampler(BaseSampler):
    """An estimator based on opflow.

    Why? Hopefully lot's of speed.
    """

    def __init__(
        self,
        circuits: Iterable[QuantumCircuit] | QuantumCircuit | None = None,
        parameters: Iterable[Iterable[Parameter]] | None = None,
        options: dict | None = None,
        # expectation_converter: ExpectationBase | None = None,
        backend: QuantumInstance | Backend | None = None,
    ) -> None:
        """
        Args:
            circuits: Not supported.
            parameters: Not supported.
            options: Not supported.
            expectation_converter: An expectation converter.
            backend: The backend or quantum instance to execute circuits.
        """
        for name, arg in {
            ("circuits", circuits),
            ("parameters", parameters),
            ("options", options),
        }:
            if arg is not None:
                raise ValueError(f"{name} is not supported in the Opflow primitive.")

        super().__init__(circuits, parameters, options)

        if backend is None:
            raise ValueError("backend is required.")

        if not isinstance(backend, QuantumInstance):
            backend = QuantumInstance(backend)

        self.sampler = CircuitSampler(backend)

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> Job:
        to_sample = {}

        for i, circuit in enumerate(circuits):
            circuit.remove_final_measurements()
            key = _circuit_key(circuit)

            # generate a dictionary with the expectation as key,
            # and as value a tuple of (parameters, [index1, index2, ...], [values1, values2, ...])
            if key not in to_sample.keys():
                to_sample[key] = (circuit, [i], [parameter_values[i]])
            else:
                to_sample[key][1].append(i)
                to_sample[key][2].append(parameter_values[i])

        job = PrimitiveJob(self._evaluate, to_sample, len(circuits))
        job.submit()

        return job

    def _evaluate(self, grouped, num_results) -> SamplerResult:
        values = [None] * num_results
        metadata = [None] * num_results
        for _, (circuit, indices, values_list) in grouped.items():
            transposed = np.array(values_list).T.tolist()
            param_dict = dict(zip(circuit.parameters, transposed))
            sampled = self.sampler.convert(StateFn(circuit), params=param_dict)
            value = self._to_quasi_distribution(sampled.eval())

            for i, value_i in zip(indices, value):
                values[i] = value_i

        result = SamplerResult(values, metadata=metadata)
        return result

    def _call(
        self,
        circuits: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        raise RuntimeError("I don't need this mate!")

    def _to_quasi_distribution(self, values):
        if not isinstance(values, ListOp):
            values = [values]

        quasis = []
        for sparse_fn in values:
            as_dict = sparse_fn.to_dict_fn().primitive
            quasi = QuasiDistribution(
                {key: np.abs(amplitude) ** 2 for key, amplitude in as_dict.items()}
            )
            quasis.append(quasi)

        return quasis
