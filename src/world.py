import numpy as np
from market_maker import MarketMaker
from constants import *

class World:
    def __init__(self):
        self._market_maker = MarketMaker()
        
    def start():
        return
    
class State:
    def __init__(self, market_maker: MarketMaker):
        self._bitstring = None
        self._market_maker = market_maker
        self._asset_indexes = {
            "asset_1": 0,
            "asset_2": 1,
            "asset_3": 2
        }

    def update_bitstring(self):
        self._bitstring = np.zeros((3, NUM_INDICATORS))

        def compute_moving_average(price_history, n_steps):
            if n_steps > len(price_history):
                return np.mean(price_history)
            else:
                return np.mean(price_history[-n_steps:]) 

        for asset, index in self._asset_indexes.items():
            price = self._market_maker.get_price(asset)
            price_history = self._market_maker.get_price_history(asset)
            dividend = self._market_maker.get_dividend(asset)

            self._bitstring[index, 0] = INTEREST_RATE * price / dividend > 0.75
            self._bitstring[index, 1] = INTEREST_RATE * price / dividend > 1
            self._bitstring[index, 2] = INTEREST_RATE * price / dividend > 1.25
            self._bitstring[index, 3] = price > compute_moving_average(price_history, 10)
            self._bitstring[index, 4] = price > compute_moving_average(price_history, 100)
            self._bitstring[index, 5] = price > compute_moving_average(price_history, 500)
            self._bitstring[index, 6] = True


            
            