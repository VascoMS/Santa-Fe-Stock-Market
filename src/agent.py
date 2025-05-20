# agent.py

import numpy as np
from market_maker import MarketMaker
from constants import *
from predictor import Predictor
from typing import Any, Dict, List, Tuple

class Agent:
    def __init__(self, id: str, cash: float, asset_indexes: Dict[str, int]):
        self._id = id
        # holdings in each of the three assets
        self._portfolio = {"asset_1": 0, "asset_2": 0, "asset_3": 0}
        # cash on hand
        self._cash = cash
        # desired holdings / demand each period
        self._demand = {"asset_1": 0, "asset_2": 0, "asset_3": 0}

        self._asset_indexes = asset_indexes

        self._latest_observation = None

        self._latest_activated_predictors: Dict[str, Predictor] = {"asset_1": None, "asset_2": None, "asset_3": None}

        # create a pool of predictors per asset
        self._predictors: Dict[str, List[Predictor]] = {}
        for asset in self._portfolio:
            self._predictors[asset] = [
                Predictor(asset, self._asset_indexes[asset]) for _ in range(NUM_PREDICTORS)
            ]
        

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Process and store an observation of the market state.
        
        Parameters:
        - observation: Dictionary containing:
            - bitstring: Binary array of market indicators
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
        
        bitstring = self._latest_observation["bitstring"]
        prices = self._latest_observation["prices"]
        uncleared_assets = self._latest_observation["uncleared_assets"]

        # Get predictions (vector) and covariance matrix
        prediction_vec, covariance_matrix = self._get_predictions_and_covariance_matrix(bitstring)

        price_vec = np.array([prices[asset] for asset in self._portfolio])

        # Compute expected excess returns (μ vector)
        mu = prediction_vec - price_vec * (1 + INTEREST_RATE)

        # Compute CARA-optimal holdings: h = (1/λ) * Σ⁻¹ * μ
        try:
            inv_cov = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            # Handle singular matrix gracefully
            inv_cov = np.linalg.pinv(covariance_matrix)

        h_star = (1 / RISK_AVERSION) * inv_cov.dot(mu)

        # Prepare output
        demands_and_slope = {}
        for asset, idx in self._asset_indexes.items():
            if(asset not in uncleared_assets):
                continue
            qty = int(np.round(h_star[idx]))
            # Record for portfolio update
            self._demand[asset] = qty
            #slope = (best_p.get_parameter_a() - (1 + INTEREST_RATE)) / (RISK_AVERSION * variance) # dh/dp
            slope = 0
            demands_and_slope[asset] = (qty, slope)
            
        return demands_and_slope
    
    def _get_predictions_and_covariance_matrix(self, bitstring: List[bool]) -> Tuple:
        # Pick predictors that "fire" on the current bitstring and predict the next price + dividend as well as the covariance matrix
        predictions = []
        covariance_matrix = np.zeros((NUM_ASSETS, NUM_ASSETS))
        for asset in self._portfolio:
            idx = self._asset_indexes[asset]
            active_predictors = [
                p for p in self._predictors[asset]
                if p.matches(bitstring[idx])
            ]
            if not active_predictors:
                predictions[asset] = None
            best_p = min(active_predictors, key=lambda p: p.get_variance())

            covariance_matrix[idx] = best_p.get_covariances()

            self._latest_activated_predictors[asset] = best_p

            predictions.append(best_p.predict(
                self._latest_observation["prices"][asset],
                self._latest_observation["dividends"][asset]
            ))
        return np.array(predictions), covariance_matrix

    
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
        predictor_errors = []
        
        # Calculate the error for each predictor
        for asset in self._portfolio:
            true_price = prices[asset]
            true_dividend = dividends[asset]
            predictor = self._latest_activated_predictors[asset]
            error = predictor.calc_error(true_price, true_dividend)
            predictor_errors.append(error)

        # Update each predictor's performance
        for asset in self._portfolio:
            self._latest_activated_predictors[asset].update(predictor_errors)
            # Occasionally evolve predictors
            if np.random.rand() < 1 / GENETIC_EXPLORATION_PARAMETER:
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
