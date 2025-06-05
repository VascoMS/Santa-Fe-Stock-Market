import numpy as np
from constants import *

SEED = 42
np.random.seed(SEED)

class Predictor:
    def __init__(self, asset_name: str):
        self._asset_name = asset_name
        self._a = np.random.uniform(0.7, 1.2) if MODE != 1 else HREE_A
        self._b = np.random.uniform(-10, 19.002) if MODE != 1 else HREE_B
        self._variance = INITIAL_DEFAULT_VARIANCE
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

    @classmethod
    def generate_default_predictor(cls, asset_name: str):
        predictor = Predictor(asset_name)
        predictor._a = HREE_A
        predictor._b = HREE_B
        predictor._condition_string = ['#'] * NUM_INDICATORS
        return predictor

    @classmethod
    def load_from_dict(cls, data: dict):
        predictor = Predictor("asset_1")
        predictor._a = data["_a"]
        predictor._b = data["_b"]
        predictor._variance = data["_variance"]
        predictor._condition_string = data["_condition_string"]
        return predictor

    def predict(self, current_price: float, current_dividend: float) -> float:
        prediction = self._a * (current_price + current_dividend) + self._b
        self._last_prediction = prediction
        return prediction

    def update(self, true_price: float, true_dividend: float):
        if self._last_prediction is None:
            return
        squared_error = (self._last_prediction - (true_price + true_dividend)) ** 2
        self._variance = (1 - LAMBDA) * self._variance + LAMBDA * squared_error

    def mutate_params(self):
        parameter_mutation_type = np.random.choice(["add", "sample", "nothing"], p =[0.2, 0.2, 0.6])

        if parameter_mutation_type == "nothing":
            return
        elif parameter_mutation_type == "add":
            a_range_size = 1.2 - 0.7
            b_range_size = 19.002 - (-10)
            change_size_a = a_range_size * 0.0005 # Mutation rate for parameter a 0.05 % over the initial range
            change_size_b = b_range_size * 0.0005 # Mutation rate for parameter b 0.05 % over the initial range
            self._a += np.random.uniform(-change_size_a, change_size_a)
            self._b += np.random.uniform(-change_size_b, change_size_b)
        elif parameter_mutation_type == "sample":
            self._a = np.random.uniform(0.7, 1.2)
            self._b = np.random.uniform(-10, 19.002)
    

    def clone(self):
        clone = Predictor(self._asset_name)
        clone._a = self._a
        clone._b = self._b
        clone._variance = self._variance
        clone._condition_string = self._condition_string.copy()
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
    
    def set_parameter_a(self, a: float):
        self._a = a

    def set_parameter_b(self, b: float):
        self._b = b
    
    def set_variance(self, variance: float):
        self._variance = variance
    
    def set_condition_string(self, condition_string: list):
        self._condition_string = condition_string
    
    def set_default_condition_string(self):
        self._condition_string = ['#'] * NUM_INDICATORS

    def to_dict(self):
        return {
            "asset_name": self._asset_name,
            "a": self._a,
            "b": self._b,
            "variance": self._variance,
            "condition_string": self._condition_string
        }
