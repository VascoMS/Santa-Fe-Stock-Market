from market_maker import MarketMaker, Asset
from constants import *
from predictor import Predictor

class Agent:
    def __init__(self, id: str, cash: float, market_maker: MarketMaker):
        self._id = id
        self._portfolio = {"asset_1": 0, "asset_2": 0, "asset_3": 0}
        self._cash = cash
        self._demand = {"asset_1": 0, "asset_2": 0, "asset_3": 0}
        self._market_maker = market_maker
        self._predictors = {}
        for asset in self._portfolio:
            self._predictors[asset] = []
            for _ in range(NUM_PREDICTORS):
                self._predictors[asset].append(Predictor(asset))

    def compute_wealth(self):
        return self._cash + sum([self._portfolio[asset] * self._market_maker.get_price(asset) for asset in self._portfolio])

    def update_cash(self):
        self._cash = self._cash * (1 + INTEREST_RATE) + sum([self._portfolio[asset] * self._market_maker.get_dividend(asset) for asset in self._portfolio])

    def update_portfolio(self):
        self._portfolio = self._demand.copy()