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
            self._state.update_bitstring()
            bits = self._state.bitstring  # shape (3, NUM_INDICATORS)

            # 3. Each agent submits demand for each asset
            for agent in self._agents:
                agent.submit_orders(self._state.bitstring, self._state._asset_indexes)

            # 4. Market clearing: determine new prices & update dividends
            for asset in self._market_maker._assets:
                new_price, cleared = self._market_maker.determine_price(asset)
                # Update price and dividend
                self._market_maker._assets[asset].set_price(new_price)
                self._market_maker._assets[asset].update_dividend()

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


            
            