# predictor.py
import random
import numpy as np
from collections import deque
from constants import NUM_INDICATORS

class Predictor:
    def __init__(self, asset_name: str, memory_length: int = 50):
        self._asset_name = asset_name
        self._parameters = {
            "a": np.random.uniform(0.4, 0.6),
            "b": np.random.uniform(0.4, 0.6),
        }
        self._error_history = deque(maxlen=memory_length)
        self._variance = 1.0
        self._last_prediction = None

        # Ternary string: each character is '0', '1', or '#'
        self._condition_string = [random.choice(['0', '1', '#']) for _ in range(NUM_INDICATORS)]

    def matches(self, bitstring: np.ndarray) -> bool:
        """Check if this predictor is activated given the world state (bitstring)"""
        for cond_bit, world_bit in zip(self._condition, bitstring):
            if cond_bit == '#':
                continue
            if cond_bit == '1' and not world_bit:
                return False
            if cond_bit == '0' and world_bit:
                return False
        return True
    
    def predict(self, current_price: float, current_dividend: float) -> float:
        prediction = self._parameters["a"] * (current_price + current_dividend) + self._parameters["b"]
        self._last_prediction = prediction
        return prediction

    def update(self, true_price: float, true_dividend: float):
        if self._last_prediction is None:
            return
        error = (self._last_prediction - (true_price + true_dividend)) ** 2
        self._error_history.append(error)
        self._variance = np.mean(self._error_history) if self._error_history else 1.0

    def mutate(self, mutation_rate: float = 0.05):
        # Mutate numerical parameters
        self._parameters["a"] += np.random.normal(0, mutation_rate)
        self._parameters["b"] += np.random.normal(0, mutation_rate)
        self._parameters["a"] = np.clip(self._parameters["a"], -2, 2)
        self._parameters["b"] = np.clip(self._parameters["b"], -2, 2)

        # Mutate condition bits (10% chance per bit)
        for i in range(len(self._condition)):
            if random.random() < 0.1:
                self._condition[i] = random.choice(['0', '1', '#'])

    def clone(self):
        clone = Predictor(self._asset_name)
        clone._parameters = self._parameters.copy()
        return clone

    def get_variance(self):
        return self._variance

    def get_asset_name(self):
        return self._asset_name
