import time
import numpy as np

from qiskit.circuit.library import EfficientSU2
from qiskit_aer.backends import AerSimulator
from qiskit.opflow import PauliSumOp, PauliExpectation
from qiskit.utils import QuantumInstance

from patchwork.primitives import OpflowEstimator

backend = AerSimulator(shots=1024)
estimator = OpflowEstimator(expectation_converter=PauliExpectation(), backend=backend)

circuit = EfficientSU2(num_qubits=3, reps=1, entanglement="pairwise")
obs = PauliSumOp.from_list([("Z" * circuit.num_qubits, 1)])

job1 = estimator.run([circuit], [obs], [np.zeros(circuit.num_parameters)])
job2 = estimator.run([circuit], [obs], [np.ones(circuit.num_parameters)])
results = [job1.result(), job2.result()]
print(results)
