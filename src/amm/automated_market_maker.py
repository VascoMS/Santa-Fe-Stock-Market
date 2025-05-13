from data import QUOTE_ASSET, OrderType
from pool import Pool
from typing import Dict

class AutomatedMarketMaker:
    def __init__(self):
        self.pools: Dict[str, Pool] = {}
    
    def create_pool(self, asset: str, size_usd: str, price: float, fee: float) -> None:
        if asset in self.pools:
            raise ValueError(f"Pool already exists for {asset}")
        pool = Pool(asset, QUOTE_ASSET, fee)
        self.pools[asset] = pool
        pool.add_liquidity(size_usd/price, size_usd)        

    def trade(self, type: OrderType, asset: str, amount_in: float, token_out: str) -> float:
        pool = self.pools.get(asset)
        if not pool:
            raise ValueError(f"No pool found for {asset}")
        if type == OrderType.BUY:
            result = pool.swap(amount_in, QUOTE_ASSET)
        else:
            result = pool.swap(amount_in, asset)
        return result

    def get_asset_price(self, asset: str) -> float:
        return self.pools[asset].get_price() if asset in self.pools else None
    
    def get_all_asset_prices(self) -> Dict[str, float]:
        return {asset: pool.get_price() for asset, pool in self.pools.items()}

