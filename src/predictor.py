import random
import numpy as np
from constants import NUM_INDICATORS, LAMBDA, M, C

class Predictor:
    def __init__(self, asset_name: str):
        self._asset_name = asset_name
        self._a = np.random.uniform(0.7, 1.2)
        self._b = np.random.uniform(-10, 19.002)
        self._variance = 4.0
        self._last_prediction = None

        self._condition_string = [np.random.choice(['0', '1', '#'], p=[0.1, 0.1, 0.8]) for _ in range(NUM_INDICATORS)]

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
        if random.random() <= 0.03:
            self._a += np.random.normal(-mutation_rate, mutation_rate)
            self._b += np.random.normal(-mutation_rate, mutation_rate)
        for i in range(len(self._condition_string) - 2):
            if random.random() < 0.03:
                self._condition_string[i] = random.choice(['0', '1', '#'])

    def clone(self):
        clone = Predictor(self._asset_name)
        clone._a = self._a
        clone._b = self._b
        return clone


    def get_variance(self):
        return self._variance
    
    def calculate_fitness(self):
        s = sum(1 for bit in self._condition_string if bit == '1' or bit == '0')
        return M - self._variance - C * s

    def get_asset_name(self):
        return self._asset_name

    def get_parameter_a(self):
        return self._a
    
    def get_parameter_b(self):
        return self._b