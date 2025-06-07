from time import sleep
from typing import List
import numpy as np
from market_maker import MarketMaker
from agent import Agent
from constants import *
import matplotlib.pyplot as plt
import pandas as pd
import os, datetime
import json
import os
from predictor import Predictor
import datetime

SEED = 42
np.random.seed(SEED)

class World:
    def __init__(self, experiment_id: int = 0, parameter_filename: str = None, pricehistory_filename: str = None):
        # Create agents with initial cash 
        self._market_maker = MarketMaker()
        if LOAD_STATE:
            price_history = pd.read_csv(pricehistory_filename, header=None).values.flatten().tolist()
            self._market_maker.get_asset("asset_1").set_price_history(price_history)
            self._market_maker.get_asset("asset_1").set_price(price_history[-1])
            saved_predictors = self.load_agent_predictors(parameter_filename)
            self._agents = [Agent(str(i), AGENT_INITIAL_CASH, saved_predictors[i]) for i in range(NUM_AGENTS)]
        else:
            self._agents = [Agent(str(i), AGENT_INITIAL_CASH) for i in range(NUM_AGENTS)]
        # State object for computing world bits
        self._state = State()

        self._experiment_id = experiment_id
    

    def load_agent_predictors(self, parameter_filename: str):
        predictor_dict = {}
        with open(parameter_filename, 'r') as f:
            saved_predictors = json.load(f)
            for i in range(NUM_AGENTS):
                agent_predictors = []
                loaded_predictors_for_agent = saved_predictors[str(i)]["asset_1"]
                for predictor in loaded_predictors_for_agent:
                    predictor = Predictor.load_from_dict(predictor)
                    agent_predictors.append(predictor)
                predictor_dict[i] = agent_predictors
        return predictor_dict
        
                        
    def _run_auctions(self):
        # 1. Launch new auctions
        self._market_maker.start_auctions()

        uncleared_assets = self._market_maker.get_uncleared_assets()
        prices = self._market_maker.get_all_prices()
        dividends = self._market_maker.get_all_dividends()
        price_histories = self._market_maker.get_all_price_histories()

        # 2. Compute current information bits
        bits = self._state.update_bitstring(prices, price_histories, dividends)     

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
                
                    
            # Run auctions for the uncleared asset and get new price
            new_prices = self._market_maker.run_auctions(uncleared_assets)

            for asset in uncleared_assets:
                prices[asset] = new_prices[asset]

            uncleared_assets = self._market_maker.get_uncleared_assets()
        
        for agent in self._agents:
            agent._auction_beginning = True

        # Update price and dividend for asset
        self._market_maker.finalize_auctions()
        self._market_maker.update_dividends()

    def run(self):
        avg_technical_bits = []
        for t in range(1, NUM_STEPS+1):
            print(f"Step {t}/{NUM_STEPS}")
            # 1. Start new auctions    
            self._run_auctions()
            # 2. Update agents
            technical_bits_per_t = 0
            for agent in self._agents:
                agent.update()
                for _, predictors in agent._predictors.items():
                    # Count technical bits used by the predictor
                    for predictor in predictors:
                        technical_bits_per_t += sum(1 for bit in predictor._condition_string[6:10] if bit == '1' or bit == '0')
            avg_technical_bits.append(technical_bits_per_t / NUM_AGENTS)
        
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = f"results/{time}_{REGIME}/experiment_{self._experiment_id}"

        self.save_metrics_and_plots(avg_technical_bits, folder)

        if SAVE_PREDICTORS:
            self.save_predictors(folder)
            
    
    def save_predictors(self, folder: str):
        predictors = {}
        for agent in self._agents:
            # Create directory for saving predictors
            os.makedirs(folder, exist_ok=True)

            # Save this agent's predictors
            agent_predictors = {}
            for asset, predictors_list in agent._predictors.items():
                agent_predictors[asset] = []
                for predictor in predictors_list:
                    predictor_data = predictor.to_dict()
                    agent_predictors[asset].append(predictor_data)
            predictors[agent._id] = agent_predictors
        # Save to JSON
        with open(f"{folder}/predictors.json", 'w') as f:
            json.dump(predictors, f, indent=4)
        
        

    def save_metrics_and_plots(self, avg_technical_bits: List[int], folder: str, steps: int = NUM_STEPS):
        os.makedirs(folder, exist_ok=True)
        os.makedirs(f"{folder}/plots", exist_ok=True)
        os.makedirs(f"{folder}/data", exist_ok=True)

        price_history = self._market_maker.get_all_price_histories()["asset_1"]
        self._plot_prices(price_history, f'{folder}/plots/{REGIME}_{steps}_price_history.png')
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
        import matplotlib.pyplot as plt
        plt.figure()  # Start a new figure
        plt.plot(range(len(price_history)), price_history)
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.title(f'Asset Price History')
        plt.savefig(filename)  # Save with a filename
        plt.close()  # Close the figure to free memory
    
class State:
    def __init__(self):
        self._asset_indexes = {
            "asset_1": 0,
        }
        self._bitstring = np.zeros((NUM_ASSETS, NUM_INDICATORS))
    
    def update_bitstring(self, prices, price_histories, dividends):
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