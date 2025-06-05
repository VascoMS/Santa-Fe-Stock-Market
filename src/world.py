from typing import List
import numpy as np
from market_maker import MarketMaker
from agent import Agent
from constants import *
import matplotlib.pyplot as plt
import pandas as pd
import os, datetime

SEED = 42
np.random.seed(SEED)

class World:
    def __init__(self, experiment_id: int = 0):
        # Initialize market maker and agents
        self._market_maker = MarketMaker()
        # Create agents with an initial cash endowment
        self._agents = [Agent(str(i), AGENT_INITIAL_CASH) for i in range(NUM_AGENTS)]
        # State object for computing world bits
        self._state = State()

        self._experiment_id = experiment_id


    def _run_auctions(self):
        # 1. Launch new auctions
        self._market_maker.start_auctions()

        uncleared_assets = self._market_maker.get_uncleared_assets()
        prices = self._market_maker.get_all_prices()
        dividends = self._market_maker.get_all_dividends()
        price_histories = self._market_maker.get_all_price_histories()

        # 2. Compute current information bits
        bits = self._state.update_bitstring(prices, price_histories, dividends) # shape (3, NUM_INDICATORS)    

        i = 0
        
        while uncleared_assets and i < MAX_AUCTION_ITERATIONS:
            i += 1
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
                agent._auction_beginning = False
                
                # Submit demands to market maker
                for asset, (demand, slope) in demands_and_slope.items():
                    self._market_maker.add_demand(asset, demand, slope)
                
                    
            # Run auctions for all uncleared assets and get new prices
            new_prices = self._market_maker.run_auctions(uncleared_assets)

            for asset in uncleared_assets:
                prices[asset] = new_prices[asset]

            uncleared_assets = self._market_maker.get_uncleared_assets()
        
        for agent in self._agents:
            agent._auction_beginning = True
        #print(f"Auctions cleared after {i} iterations.")

        # Update prices and dividends for all assets
        #print(f"Market cleared: {len(uncleared_assets) == 0}")
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
        avg_technical_bits = []
        for t in range(1, NUM_STEPS+1):
            print(f"Step {t}/{NUM_STEPS}")
            # 1. Start new auctions    
            self._run_auctions()
            # 2. Update agents
            technical_bits_per_t = 0
            for agent in self._agents:
                agent.update()
                for asset, predictors in agent._predictors.items():
                    # Count technical bits used by the predictor
                    for predictor in predictors:
                        technical_bits_per_t += sum(1 for bit in predictor._condition_string[6:10] if bit == '1' or bit == '0')
            avg_technical_bits.append(technical_bits_per_t / NUM_AGENTS)
        self.save_metrics_and_plots(avg_technical_bits)
    

    def save_metrics_and_plots(self, avg_technical_bits: List[int], steps: int = NUM_STEPS):
        """
        Save metrics and plots for the simulation.
        This method can be extended to save more detailed metrics as needed.
        """
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = f"results/{time}/experiment_{self._experiment_id}"
        os.makedirs(folder, exist_ok=True)
        os.makedirs(f"{folder}/plots", exist_ok=True)
        os.makedirs(f"{folder}/data", exist_ok=True)

        price_history = self._market_maker.get_all_price_histories()["asset_1"]
        self._plot_prices(price_history, f'{folder}/plots/Experiment_{self._experiment_id}_{REGIME}_{steps}_price_history.png')
        x_values = range(len(avg_technical_bits))
        self._plot_values(avg_technical_bits, x_values, "Average Technical Bits Used Per Step", "Time Step", "Average Technical Bits", f"{folder}/plots/Experiment_{self._experiment_id}_{REGIME}_{steps}_avg_technical_bits.png")
        pd.Series(price_history).to_csv(f"{folder}/data/{REGIME}_{steps}_price_history.csv", index=False, header=False)

        combined_volume = [sum(vol) for vol in zip(*[agent._trading_volumes for agent in self._agents])]
        pd.Series(combined_volume).to_csv(f"{folder}/data/{REGIME}_{steps}_combined_volume.csv", index=False, header=False)

    def _plot_values(self, y_values, x_values, title, xlabel, ylabel, filename):
        plt.figure(figsize=(10, 5))

        # Subsample every 4% of the data
        step = max(1, int(len(x_values) * 0.04))
        x_sub = x_values[::step]
        y_sub = y_values[::step]

        plt.plot(x_sub, y_sub, marker='o', linestyle='-', linewidth=1)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.savefig(filename)
        plt.close()

    
    def _plot_prices(self, price_history: list, filename: str):
        """
        Plot the price history of all assets.
        
        Parameters:
        - prices: Dictionary mapping asset IDs to lists of historical prices
        """
        import matplotlib.pyplot as plt
        offset = 500 
        plt.figure()  # Start a new figure
        plt.plot(range(offset, len(price_history)), price_history[offset:])
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.title(f'Asset Price History')
        plt.savefig(filename)  # Save with a filename
        plt.close()  # Close the figure to free memory
    
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

            for i in range(len(fundamental_thresholds)): # Fundamental indicators
                self._bitstring[index, i] = INTEREST_RATE * price / dividend > fundamental_thresholds[i]
            
            for i in range(len(ma_windows)): # Technical indicators
                self._bitstring[index, i + len(fundamental_thresholds)] = price > compute_moving_average(price_history, ma_windows[i])

            self._bitstring[index, -1] = True
            self._bitstring[index, -2] = False
        
        return self._bitstring

            
            