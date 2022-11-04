# Purpose

This repo provides `qiskit.opflow` based primitives until the simulation-based primitives are up-to-speed and as performant as the previous simulators.
With these primitives, researchers & co. can already update their code to the primitive workflow and easily switch to a session execution model for backend, 
without sacrificing performance.

### Did you say performance?

Yes, check out the `examples/benchmark.py` which compares 500 expectation value calculations on a shallow 10-qubit circuit. The results are:
```
Opflow primitive:
1.928851842880249  # <-- in this repo
Reference primitive:
3.620875120162964
Aer primitive:
91.24100494384766
```
