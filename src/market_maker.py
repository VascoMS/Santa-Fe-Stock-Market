from collections import deque
import numpy as np
from typing import Dict, Tuple, List
from constants import *

class Asset:
    def __init__(self, initial_dividend: float, rho: float, alpha: float, supply: int):
        self._dividend = initial_dividend
        self._dividend_mean = initial_dividend
        self._prev_dividend_delta = self._dividend - self._dividend_mean
        self._price = initial_dividend / INTEREST_RATE # setting initial price to fair price of d(t)/r 
        self._price_history = deque([self._price]) # Store last 100 prices
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
    
    def get_price_history(self) -> List[float]:
        return self._price_history.copy()
    
    def get_supply(self) -> int:
        return self._supply
    
    def set_price(self, price: float):
        self._price = price
        self._price_history.append(price)

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
        cleared = self.cleared()
        if not cleared:
            self._price = self._price - self._k * (delta/self._slope)
            self._demand = 0
        return self._price, cleared
    
    def clear_demand(self):
        self._demand = 0
        self._slope = 0
    
    def cleared(self) -> bool:
        return self._demand - self._supply < 1e-2

class MarketMaker:
    K = 0.001

    def __init__(self):
        self._assets: Dict[str, Asset] = self._create_assets()
        self._auctions: Dict[str, Auction] = dict()

    def _create_assets(self):
        return {
        "asset_1": Asset(
            initial_dividend=ASSET_1_INITIAL_DIVIDEND,
            rho=ASSET_1_RHO,
            alpha=ASSET_1_ALPHA,
            supply=ASSET_1_SUPPLY
        ),
        "asset_2": Asset(
            initial_dividend=ASSET_2_INITIAL_DIVIDEND,
            rho=ASSET_2_RHO,
            alpha=ASSET_2_ALPHA,
            supply=ASSET_2_SUPPLY
        ),
        "asset_3": Asset(
            initial_dividend=ASSET_3_INITIAL_DIVIDEND,
            rho=ASSET_3_RHO,
            alpha=ASSET_3_ALPHA,
            supply=ASSET_3_SUPPLY
        )
    }
    
    def add_demand(self, asset: str, amount: int, slope:int):
        auction = self._auctions[asset]
        auction.add_demand(amount, slope)

    
    def run_auctions(self, assets: List[str]) -> Tuple[int, bool]:
        auction_prices = dict()
        for asset in assets:
            auction = self._auctions[asset]
            if auction and not self._auctions[asset].cleared():
                auction.clear_demand()
                price, _ = self._auctions[asset].determine_price()
                auction_prices[asset] = price
        return auction_prices
    
    def finalize_auctions(self):
        for asset_id, asset in self._assets.items():
            auction = self._auctions[asset_id]
            if auction and auction.cleared():
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
        for asset in self._assets.values():
            asset.update_dividend()
    
    def clear_auctions(self, assets: List[str]):
        for asset in assets:
            self._auctions[asset].clear_demand()