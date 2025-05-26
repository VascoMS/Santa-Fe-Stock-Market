# agent.py

import numpy as np
from market_maker import MarketMaker
from constants import *
from predictor import Predictor
from typing import Any, Dict, List, Tuple
import random

class Agent:
    def __init__(self, id: str, cash: float):
        self._id = id
        random.random()
        # holdings in each of the three assets
        self._portfolio = {"asset_1": 0, "asset_2": 0, "asset_3": 0}
        # cash on hand
        self._cash = cash
        # desired holdings / demand each period
        self._demand = {"asset_1": 0, "asset_2": 0, "asset_3": 0}

        self._latest_observation = None

        # create a pool of predictors per asset
        self._predictors: Dict[str, List[Predictor]] = {}
        for asset in self._portfolio:
            self._predictors[asset] = [
                Predictor(asset) for _ in range(NUM_PREDICTORS)
            ]
        self._auction_beginning = True
        self._expected = 0
        self._latest_predictor = None

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Process and store an observation of the market state.
        
        Parameters:
        - observation: Dictionary containing:
            - bitstring: Binary array of market indicators
            - asset_indexes: Mapping of asset IDs to their indexes in the bitstring
            - prices: Current prices of all assets
            - dividends: Current dividends of all assets
            - uncleared_assets: List of assets with uncleared markets
        """
        self._latest_observation = observation

    def act(self) -> Dict[str, Tuple[int, float]]:
        """
        Calculate demands based on the latest observation.
        Returns a dictionary of asset_id -> (quantity, price_slope).
        """
        if not self._latest_observation:
            return {}
            
        bitstring = self._latest_observation["bitstring"]
        asset_indexes = self._latest_observation["asset_indexes"]
        prices = self._latest_observation["prices"]
        dividends = self._latest_observation["dividends"]
        uncleared_assets = self._latest_observation.get("uncleared_assets", list(asset_indexes.keys()))
        
        demands_and_slope = {}
        for asset in uncleared_assets:
            idx = asset_indexes[asset]
            price = prices[asset]
            dividend = dividends[asset]

            if self._auction_beginning:
                # Pick predictors that "fire" on the current bitstring
                active_predictors = [
                    p for p in self._predictors[asset]
                    if p.matches(bitstring[idx])
                ]
                
                
                if not active_predictors:
                    # No signal ⇒ no position
                    self._demand[asset] = 0
                    demands_and_slope[asset] = (0, 0)
                    continue

                # Choose the most precise predictor
                best_p = min(active_predictors, key=lambda p: p.get_variance())
                # One-step ahead forecast of total payout
                self._expected = best_p.predict(price, dividend)
                self._latest_predictor = best_p

                print(f"Agent {self._id} is computing new expected value and variance for asset {asset}: Expected = {self._expected}, Variance = {self._latest_predictor.get_variance()}")

            # CARA-optimal target shares
            target_h = (self._expected - price * (1 + INTEREST_RATE)) / (
                RISK_AVERSION * self._latest_predictor.get_variance()
            )
            qty = self._bound_demand(np.round(target_h, 2), self._portfolio[asset], price)
            print(f"Agent {self._id} - Asset: {asset}, Expected: {self._expected}, Price: {price}, Target_h: ({self._expected} - {price} * (1 + {INTEREST_RATE})) / ({RISK_AVERSION} * {self._latest_predictor.get_variance()}) = {target_h}, Demand: {qty}")

            # Record for portfolio update
            self._demand[asset] = qty
            # Submit to auction 
            slope = (self._latest_predictor.get_parameter_a() - (1 + INTEREST_RATE)) / (RISK_AVERSION * self._latest_predictor.get_variance()) # dh/dp
            demands_and_slope[asset] = (qty, slope)
            
        return demands_and_slope
    
    def _bound_demand(self, demand, current_holding, price) -> None:
        return min(max(demand, 0), self._cash/price + current_holding)
    
    def compute_wealth(self) -> float:
        """Calculate total wealth = cash + market value of all holdings."""
        if not self._latest_observation:
            return self._cash
            
        prices = self._latest_observation["prices"]
        
        portfolio_value = sum(
            self._portfolio[asset] * prices[asset]
            for asset in self._portfolio if asset in prices
        )
        
        return self._cash + portfolio_value

    def _update_cash(self) -> None:
        """Roll cash forward by interest and collect dividends."""
        if not self._latest_observation:
            return
            
        dividends = self._latest_observation["dividends"]
        
        # Interest on cash
        self._cash *= (1 + INTEREST_RATE)
        
        # Dividends on holdings
        self._cash += sum(
            self._portfolio[asset] * dividends[asset]
            for asset in self._portfolio if asset in dividends
        )

    def _update_portfolio(self):
        """Re‐balance portfolio to match last period’s submitted demand."""
        asset_deltas = {asset: self._demand[asset] - self._portfolio[asset] for asset in self._portfolio}
        self._cash -= sum(
            asset_deltas[asset] * self._latest_observation["prices"][asset]
            for asset in self._portfolio if asset in self._latest_observation["prices"]
        )
        self._portfolio = self._demand.copy()

    def _update_predictors(self) -> None:
        """
        Update each predictor's performance using the true price and dividend.
        Optionally: mutate or run GA here at rate 1/K.
        """
        if not self._latest_observation:
            return
            
        prices = self._latest_observation["prices"]
        dividends = self._latest_observation["dividends"]
        
        for asset in self._portfolio:
            if asset in prices and asset in dividends:
                true_price = prices[asset]
                true_dividend = dividends[asset]
                
                for predictor in self._predictors[asset]:
                    predictor.update(true_price, true_dividend)
                    
                # Occasionally evolve predictors
                if random.random() < (1.0 / GENETIC_EXPLORATION_PARAMETER):
                    self._evolve_predictors(asset)
    
    def update(self):
        """
        Update the agent's state.
        This method is called at each time step to update the agent's state.
        """
        # Update cash and portfolio
        self._update_cash()
        self._update_portfolio()
        
        # Update predictors
        self._update_predictors()
            
    def _evolve_predictors(self, asset: str) -> None:
        """
        Run a genetic algorithm to evolve the predictors for a specific asset.
        Replaces the worst 20% (highest variance) with mutated clones of the top 20% (lowest variance).
        """
        # Sort predictors by variance (ascending: low to high)
        sorted_predictors = sorted(
            self._predictors[asset],
            key=lambda p: p.get_variance()
        )

        num_to_replace = int(0.2 * len(sorted_predictors))

        # Select top 20% (lowest variance)
        best = sorted_predictors[:num_to_replace]

        # Select bottom 20% (highest variance)
        worst = sorted_predictors[-num_to_replace:]

        # Create mutated clones of the best predictors
        mutated_clones = []
        for predictor in best:
            clone = predictor.clone()
            clone.mutate()
            mutated_clones.append(clone)

        # Remove the worst predictors
        self._predictors[asset] = [
            p for p in sorted_predictors if p not in worst
        ] + mutated_clones
