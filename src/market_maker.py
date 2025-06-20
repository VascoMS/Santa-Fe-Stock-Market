from collections import deque
import numpy as np
from typing import Dict, Tuple, List
from constants import *
from math import sqrt

SEED = 42
np.random.seed(SEED)

class Asset:
    def __init__(self, initial_dividend: float, rho: float, supply: int):
        self._dividend = initial_dividend
        self._dividend_mean = initial_dividend
        self._prev_dividend_delta = self._dividend - self._dividend_mean
        self._price = initial_dividend / INTEREST_RATE
        self._price_history = deque([self._price]) 
        self._rho = rho
        self._supply = supply

    def update_dividend(self) -> float:
        self._dividend = self._dividend_mean + self._rho*(self._dividend - self._dividend_mean) + np.random.normal(0, sqrt(ASSET_1_ERROR_VARIANCE))
        return self._dividend
    
    def get_dividend(self) -> float:
        return self._dividend
        
    def get_price(self) -> float:
        return self._price
    
    def get_price_history(self) -> List[float]:
        return list(self._price_history.copy())
    
    def get_supply(self) -> int:
        return self._supply
    
    def set_price(self, price: float):
        self._price = price
        self._price_history.append(price)

    def set_price_history(self, price_history: List[float]):
        self._price_history = deque(price_history)

class Auction:
    def __init__(self, asset: Asset, k: float):
        self._price = asset.get_price()
        self._supply = asset.get_supply()
        self._demand = 0
        self._slope = 0
        self._cleared = False
        self._k = k
    
    def add_demand(self, amount: int, slope: int):
        self._demand += amount
        self._slope += slope
    
    def determine_price(self) -> int:
        delta = self._demand - self._supply
        self._cleared = abs(delta) <= 0.01
        if not self._cleared:
            if self._slope != 0:
                self._price = min(max(0.01, self._price - self._k * (delta/self._slope)), 500)
            else:
                self._price = min(max(0.01, self._price + self._k*delta), 500)
            self._demand = 0
            self._slope = 0
        return self._price
    
    def cleared(self) -> bool:
        return self._cleared

class MarketMaker:
    K = 0.1

    def __init__(self):
        self._assets: Dict[str, Asset] = self._create_assets()
        self._auctions: Dict[str, Auction] = dict()

    def _create_assets(self):
        return {
        "asset_1": Asset(
            initial_dividend=ASSET_1_INITIAL_DIVIDEND,
            rho=ASSET_1_RHO,
            supply=ASSET_1_SUPPLY
        ),
    }
    
    def add_demand(self, asset: str, amount: int, slope:int):
        auction = self._auctions[asset]
        auction.add_demand(amount, slope)

    
    def run_auctions(self, assets: List[str]) -> Tuple[int, bool]:
        auction_prices = dict()
        for asset in assets:
            auction = self._auctions[asset]
            if auction and not self._auctions[asset].cleared():
                price = self._auctions[asset].determine_price()
                auction_prices[asset] = price
        return auction_prices
    
    def finalize_auctions(self):
        for asset_id, asset in self._assets.items():
            auction = self._auctions[asset_id]
            if auction:
                asset.set_price(auction._price)
            
    def start_auctions(self):
        for asset_id, asset in self._assets.items():
            self._auctions[asset_id] = Auction(asset, self.K)
    
    def get_price(self, asset: str) -> float:
        return self._assets[asset].get_price()
    
    def get_price_history(self, asset: str) -> List[float]:
        return self._assets[asset].get_price_history()
    
    def get_dividend(self, asset: str) -> float:
        return self._assets[asset].get_dividend()
    
    def get_all_prices(self) -> Dict[str, float]:
        return {asset: self._assets[asset].get_price() for asset in self._assets}
    
    def get_all_dividends(self) -> Dict[str, float]:
        return {asset: self._assets[asset].get_dividend() for asset in self._assets}
    
    def get_all_prices_and_dividends(self) -> Dict[str, Tuple[float, float]]:
        return {asset: (self._assets[asset].get_price(), self._assets[asset].get_dividend()) for asset in self._assets}
    
    def get_all_price_histories(self) -> Dict[str, List[float]]:
        return {asset: self._assets[asset].get_price_history() for asset in self._assets}
    
    def get_uncleared_assets(self) -> List[str]:
        return [asset for asset, auction in self._auctions.items() if not auction.cleared()]
    
    def update_dividends(self):
        for id, asset in self._assets.items():
            asset.update_dividend()
    
    def clear_auctions(self, assets: List[str]):
        for asset in assets:
            self._auctions[asset].clear_demand()

    def get_asset(self, id: str) -> Asset:
        return self._assets[id]