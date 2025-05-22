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
        self._agents = [Agent(str(i), AGENT_INITIAL_CASH) for i in range(NUM_AGENTS)]
        # State object for computing world bits
        self._state = State()
    
    def _run_auctions(self):
        # 1. Launch new auctions
        self._market_maker.start_auctions()

        uncleared_assets = self._market_maker.get_uncleared_assets()
        prices = self._market_maker.get_all_prices()
        dividends = self._market_maker.get_all_dividends()
        price_histories = self._market_maker.get_all_price_histories()

        # 2. Compute current information bits
        bits = self._state.update_bitstring(prices, price_histories, dividends) # shape (3, NUM_INDICATORS)    

        while uncleared_assets:
            # Prepare the observation for agents
            observation = {
                "bitstring": bits,
                "asset_indexes": self._state._asset_indexes,
                "prices": prices,
                "dividends": dividends,
                "uncleared_assets": uncleared_assets
            }
            
            # 3. Agents observe and compute demands
            for agent in self._agents:
                # Provide the observation
                agent.observe(observation)
                # Get the agent's action
                demands_and_slope = agent.act()
                agent._auction_begining = False
                
                # Submit demands to market maker
                for asset, (demand, slope) in demands_and_slope.items():
                    self._market_maker.add_demand(asset, demand, slope)
                
                    
            # Run auctions for all uncleared assets and get new prices
            new_prices = self._market_maker.run_auctions(uncleared_assets)

            for asset in uncleared_assets:
                prices[asset] = new_prices[asset]
                price_histories[asset].append(new_prices[asset])

            bits = self._state.update_bitstring(prices, price_histories, dividends)

            uncleared_assets = self._market_maker.get_uncleared_assets()
        
        agent._auction_begining = True

        # Update prices and dividends for all assets
        self._market_maker.finalize_auctions()
        self._market_maker.update_dividends()

    def run(self):
        """
        Run the market for NUM_STEPS periods. Each step:
        1. Start new auctions for each asset
        2. Update world bitstrings for information set
        3. Agents make predictions & submit demands
        4. Market clears via auctions: prices and dividends update
        5. Agents update cash, portfolios, and predictor performance
        """
        for t in range(NUM_STEPS):
            # 1. Start new auctions    
            self._run_auctions()
            # 2. Update agents
            for agent in self._agents:
                agent.update()
                # Print agent portfolio and cash
                print(f"Agent {agent._id} Portfolio: {agent._portfolio}, Cash: {agent._cash}")

    
class State:
    def __init__(self):
        self._asset_indexes = {
            "asset_1": 0,
            # "asset_2": 1,
            # "asset_3": 2
        }
        self._bitstring = np.zeros((NUM_ASSETS, NUM_INDICATORS))
    
    def update_bitstring(self, prices, price_histories, dividends):
        """
        Update the bitstring based on market data passed in as parameters.
        
        Parameters:
        - prices: Dict mapping asset IDs to current prices
        - price_histories: Dict mapping asset IDs to lists of historical prices
        - dividends: Dict mapping asset IDs to current dividends
        
        Returns:
        - The updated bitstring numpy array
        """
        def compute_moving_average(price_history, n_steps):
            if n_steps > len(price_history):
                return np.mean(np.array(price_history))
            else:
                return np.mean(np.array(price_history)[-n_steps:]) 
        
        fundamental_thresholds = [0.25, 0.5, 0.75, 0.875, 1.0, 1.25]
        ma_windows = [5, 10, 100, 500]

        for asset, index in self._asset_indexes.items():
            price = prices[asset]
            price_history = price_histories[asset]
            dividend = dividends[asset]

            for i in range(len(fundamental_thresholds)):
                self._bitstring[index, i] = INTEREST_RATE * price / dividend > fundamental_thresholds[i]
            
            for i in range(len(ma_windows)):
                self._bitstring[index, i + len(fundamental_thresholds)] = price > compute_moving_average(price_history, ma_windows[i])

            self._bitstring[index, -1] = True
            self._bitstring[index, -2] = False
        
        return self._bitstring

            
            