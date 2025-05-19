import random
import numpy as np
from constants import NUM_INDICATORS, LAMBDA

class Predictor:
    def __init__(self, asset_name: str, memory_length: int = 50):
        self._asset_name = asset_name
        self._a = np.random.uniform(0.4, 0.6)
        self._b = np.random.uniform(0.4, 0.6)
        self._variance = 1.0
        self._last_prediction = None

        self._condition_string = [random.choice(['0', '1', '#']) for _ in range(NUM_INDICATORS)]

    def matches(self, bitstring: np.ndarray) -> bool:
        for cond_bit, world_bit in zip(self._condition_string, bitstring):
            if cond_bit == '#':
                continue
            if cond_bit == '1' and not world_bit:
                return False
            if cond_bit == '0' and world_bit:
                return False
        return True

    def predict(self, current_price: float, current_dividend: float) -> float:
        prediction = self._a * (current_price + current_dividend) + self._b
        self._last_prediction = prediction
        return prediction

    def update(self, true_price: float, true_dividend: float):
        if self._last_prediction is None:
            return
        squared_error = (self._last_prediction - (true_price + true_dividend)) ** 2
        self._variance = (1 - LAMBDA) * self._variance + LAMBDA * squared_error

    def mutate(self, mutation_rate: float = 0.05):
        self._a += np.random.normal(0, mutation_rate)
        self._b += np.random.normal(0, mutation_rate)
        self._a = np.clip(self._a, -2, 2)
        self._b = np.clip(self._b, -2, 2)

        for i in range(len(self._condition_string)):
            if random.random() < 0.1:
                self._condition_string[i] = random.choice(['0', '1', '#'])

    def clone(self):
        clone = Predictor(self._asset_name)
        clone._a = self._a
        clone._b = self._b
        return clone

    def get_variance(self):
        return self._variance

    def get_asset_name(self):
        return self._asset_name

    def get_parameter_a(self):
        return self._a
    
    def get_parameter_b(self):
        return self._b