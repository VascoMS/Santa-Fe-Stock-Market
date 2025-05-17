import numpy as np
from market_maker import MarketMaker
from constants import *

class World:
    def __init__(self, market_maker: MarketMaker):
        self._market_maker = market_maker

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

        for asset, index in self._asset_indexes.items():
            price = self._market_maker.get_price(asset)
            dividend = self._market_maker.get_dividend(asset)

            self._bitstring[index, 0] = INTEREST_RATE * price / dividend > 0.75
            self._bitstring[index, 1] = INTEREST_RATE * price / dividend > 1
            self._bitstring[index, 2] = INTEREST_RATE * price / dividend > 1.25
            self._bitstring[index, 3] = price > self.compute_moving_average(asset, 10)
            self._bitstring[index, 4] = price > self.compute_moving_average(asset, 100)
            self._bitstring[index, 5] = price > self.compute_moving_average(asset, 500)
            self._bitstring[index, 6] = True

    def compute_moving_average(self, asset, n_steps):
        if n_steps > len(self._market_maker.get_price_history(asset)):
            return np.mean(self._market_maker.get_price_history(asset))
        else:
            return np.mean(self._market_maker.get_price_history(asset)[-n_steps:]) 



            
            