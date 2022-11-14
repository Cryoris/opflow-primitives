import time
import numpy as np

from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.backends import AerSimulator
from qiskit.opflow import AerPauliExpectation, PauliSumOp, PauliExpectation
from qiskit.utils import QuantumInstance

from patchwork.primitives import OpflowEstimator

num_points = 500
circuit = EfficientSU2(num_qubits=10, reps=1, entanglement="pairwise")
# points = [np.random.random(circuit.num_parameters) for _ in range(num_points)]
points = [np.random.random(circuit.num_parameters)] * num_points
obs = PauliSumOp.from_list([("Z" * circuit.num_qubits, 1)])


def timeit(estimator):
    start = time.time()
    job = estimator.run(num_points * [circuit], num_points * [obs], points)
    results = job.result()
    stddev = np.std(np.real(results.values))
    print("Std:", stddev)
    return time.time() - start


reference_estimator = Estimator(options={"shots": 1024})
aer_estimator = AerEstimator(backend_options={"shots": 1024})

backend = AerSimulator(shots=1024)
qi = QuantumInstance(AerSimulator(), shots=1024)
sv_estimator = OpflowEstimator(
    expectation_converter=AerPauliExpectation(), backend=backend
)
shots_estimator = OpflowEstimator(
    expectation_converter=PauliExpectation(), backend=backend
)
qi_estimator = OpflowEstimator(expectation_converter=AerPauliExpectation(), backend=qi)

print("Opflow primitive, exact:")
print(timeit(sv_estimator))
print("Reference primitive:")
print(timeit(reference_estimator))
print("Opflow primitive, QI:")
print(timeit(qi_estimator))
print("Opflow primitive, shots:")
print(timeit(shots_estimator))
print("Aer primitive:")
print(timeit(aer_estimator))
