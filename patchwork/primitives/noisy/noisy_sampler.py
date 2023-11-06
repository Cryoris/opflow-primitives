from collections import defaultdict
import numpy as np

from qiskit.primitives import Sampler


class NoisySampler(Sampler):
    """A noisy sampler.

    Redistributes measurement results, where the probability of flipping a certain number of bits
    obeys a Poisson distribution with $\lambda$-parameter ``noise_factor``.
    """

    def __init__(self, circuits=None, parameters=None, options=None, noise_factor=None):
        if noise_factor is None:
            raise ValueError("noise_factor must be passed!")

        super().__init__(circuits, parameters, options)
        self.noise_factor = noise_factor

    def _call(self, circuits, parameter_values, **run_options):
        result = super()._call(circuits, parameter_values, **run_options)
        shots = self.options.get("shots", None)

        # remove 0 probabilities
        # to_remove = []
        # for dist in result.quasi_dists:
        #     for key, prob in dist.items():
        #         if prob == 0:
        #             to_remove.append(key)

        if shots is None:
            print("To add noise, set a finite number of shots.")
            return result

        for i, _ in enumerate(circuits):
            result.metadata[i]["noise_factor"] = self.noise_factor
            result.metadata[i]["dist"] = result.quasi_dists[i].copy()

            noisy_distribution = self.distribute_shots(result.quasi_dists[i], shots)
            result.quasi_dists[i].clear()
            result.quasi_dists[i].update(noisy_distribution)

        return result

    def distribute_shots(self, quasi_dist, shots):
        if any(value < 0 for value in quasi_dist.values()):
            raise ValueError("Cannot redistribute shots for negative probabilities.")

        noisy_measurements = defaultdict(int)
        for key, probability in quasi_dist.binary_probabilities().items():
            # sample the hamming distances to inject
            count = int(probability * shots)
            distances = np.random.poisson(self.noise_factor, size=count)

            # flip bits -- drawing with replacement as bits can be flipped twice
            positions = np.arange(len(key))
            for distance in distances:
                if distance > 0:
                    count -= 1
                    flip_locations = np.random.choice(positions, size=distance)
                    new_key = list(map(int, key))
                    for location in flip_locations:
                        new_key[location] = 1 - new_key[location]

                    new_key = "".join(map(str, new_key))
                    noisy_measurements[new_key] += 1

            noisy_measurements[key] += count

        normalized = {
            int(key, 2): count / shots
            for key, count in noisy_measurements.items()
            if count > 0
        }
        return normalized
