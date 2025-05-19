from typing import List
import numpy as np
from market_maker import MarketMaker
from agent import Agent
from constants import *

class World:
    def __init__(self):
        # Initialize market maker and agents
        self._market_maker = MarketMaker()
        # Create agents with an initial cash endowment
        initial_cash = AGENT_INITIAL_CASH  # define this in constants.py
        self._agents = [Agent(str(i), initial_cash, self._market_maker) for i in range(NUM_AGENTS)]
        # State object for computing world bits
        self._state = State(self._market_maker)

    def start(self):
        """
        Run the market for NUM_STEPS periods. Each step:
        1. Start new auctions for each asset
        2. Update world bitstrings for information set
        3. Agents make predictions & submit demands
        4. Market clears via auctions: prices and dividends update
        5. Agents update cash, portfolios, and predictor performance
        """
        for t in range(NUM_STEPS):
            # 1. Launch new auctions
            self._market_maker.start_auctions()

            # 2. Compute current information bits
            bits = self._state.update_bitstring() # shape (3, NUM_INDICATORS)    

            uncleared_assets = self._market_maker.get_uncleared_assets()
            prices = self._market_maker.get_all_prices()
            dividends = self._market_maker.get_all_dividends()
    
            while uncleared_assets:
                uncleared_asset_indexes = {asset: self._state._asset_indexes[asset] for asset in uncleared_assets}
            
                # 3. Agents compute demands
                for agent in self._agents:
                    demands_and_slope = agent.calc_demands(bits, uncleared_asset_indexes, prices, dividends)
                    for asset, (demand, slope) in demands_and_slope.items():
                        # Submit demand to market maker
                        self._market_maker.add_demand(asset, demand, slope)
                # Run auctions for all uncleared assets and get new prices
                prices = self._market_maker.run_auctions(uncleared_assets)

            # Update prices and dividends for all assets
            self._market_maker.finalize_auctions()
            self._market_maker.update_dividends()
            
            # 5. Agents update wealth, portfolios, and predictor performance
            for agent in self._agents:
                # Update cash by interest and any dividends
                agent.update_cash()
                # Move portfolio to match submitted demand
                agent.update_portfolio()
                # Update each predictor's performance
                agent.update_predictors()
    
class State:
    def __init__(self, market_maker: MarketMaker):
        self._bitstring = None
        self._market_maker = market_maker
        self._asset_indexes = {
            "asset_1": 0,
            "asset_2": 1,
            "asset_3": 2
        }
        self._bitstring = np.zeros((3, NUM_INDICATORS))
    
    def update_bitstring(self):
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
        
        return self._bitstring


            
            