import numpy as np
from typing import Dict, Tuple

class Asset:
    def __init__(self, initial_dividend: float, rho: float, alpha: float, bank_return: float, supply: int):
        self._dividend = initial_dividend
        self._dividend_mean = initial_dividend
        self._prev_dividend_delta = self._dividend - self._dividend_mean
        self._price = initial_dividend / bank_return # setting initial price to fair price of d(t)/r 
        self._price_history = [self._price]
        self._rho = rho # recommended -> 0.9 for d = 2 r = 0.02
        self._alpha = alpha # recommended -> 0.15 for d = 2 r = 0.02
        self._supply = supply

    def update_dividend(self) -> float:
        delta = self._rho * self._prev_dividend_delta + self._alpha * np.random.normal(loc=0, scale=1)
        self._prev_dividend_delta = delta
        self._dividend = self._dividend_mean + delta
        return self._dividend
    
    def get_dividend(self) -> float:
        return self._dividend
        
    def get_price(self) -> float:
        return self._price
    
    def get_price_history(self) -> float:
        return self._price_history
    
    def get_supply(self) -> int:
        return self._supply
    
    def set_price(self, price: float):
        self._price = price
        self._price_history.append(price)

class MarketMaker:
    K = 0.001

    def __init__(self, assets: Dict[str, Asset]):
        self._assets: Dict[str, Asset] = assets
        self._auctions: Dict[str, Auction] = dict()

    def add_demand(self, asset: str, amount: int, slope:int):
        auction = self._auctions[asset]
        auction.add_demand(amount, slope)
    
    def determine_price(self, asset: str) -> Tuple[int, bool]:
        return self._auctions[asset].determine_price()

    def start_auctions(self):
        for asset in self._assets.values:
            self._auctions[asset] = Auction(asset, self.K)
    
    def get_price(self, asset: str) -> float:
        return self._assets[asset].get_price()
    
    def get_price_history(self, asset: str) -> float:
        return self._assets[asset].get_price_history()
    
    def get_dividend(self, asset: str) -> float:
        return self._assets[asset].get_dividend()
    
    def get_all_prices(self) -> Dict[str, float]:
        return {asset: self._assets[asset].get_price() for asset in self._assets}
    
    def get_all_dividends(self) -> Dict[str, float]:
        return {asset: self._assets[asset].get_dividend() for asset in self._assets}

class Auction:
    def __init__(self, asset: Asset, k: float):
        self._price = asset.get_price()
        self._supply = asset.get_supply()
        self._demand = 0
        self._slope = 0
        self._k = k
    
    def add_demand(self, amount: int, slope: int):
        self._demand += amount
        self._slope += slope
    
    def determine_price(self) -> Tuple[int, bool]:
        delta = self._demand - self._supply
        cleared = delta < 1e-2
        if not cleared:
            self._price = self._price - self._k * (delta/self._slope)
            self._demand = 0
        return self._price, cleared