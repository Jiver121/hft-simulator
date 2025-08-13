"""
Options Pricing Models

This module implements various options pricing models including:
- Black-Scholes-Merton model for European options
- Binomial tree model for American options
- Monte Carlo simulation for exotic options
"""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from abc import ABC, abstractmethod
import math


class OptionsPricingModel(ABC):
    """Base class for options pricing models"""
    
    @abstractmethod
    def calculate_price(self, spot: float, strike: float, time_to_expiry: float,
                       risk_free_rate: float, volatility: float, 
                       option_type: str = 'call') -> float:
        """Calculate option price"""
        pass
    
    @abstractmethod
    def calculate_greeks(self, spot: float, strike: float, time_to_expiry: float,
                        risk_free_rate: float, volatility: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """Calculate option Greeks"""
        pass


class BlackScholesModel(OptionsPricingModel):
    """
    Black-Scholes-Merton model for European options
    
    The classic options pricing model assuming:
    - Constant volatility and risk-free rate
    - No dividends (can be extended)
    - European exercise
    - Lognormal distribution of underlying returns
    """
    
    def __init__(self, dividend_yield: float = 0.0):
        self.dividend_yield = dividend_yield
    
    def calculate_price(self, spot: float, strike: float, time_to_expiry: float,
                       risk_free_rate: float, volatility: float, 
                       option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price
        
        Args:
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            volatility: Annualized volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        if time_to_expiry <= 0:
            return max(0, self._intrinsic_value(spot, strike, option_type))
        
        d1, d2 = self._calculate_d1_d2(spot, strike, time_to_expiry, 
                                      risk_free_rate, volatility)
        
        if option_type.lower() == 'call':
            price = (spot * np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(d1) -
                    strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        else:  # put
            price = (strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
                    spot * np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(-d1))
        
        return max(0, price)
    
    def calculate_greeks(self, spot: float, strike: float, time_to_expiry: float,
                        risk_free_rate: float, volatility: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all option Greeks
        
        Returns:
            Dictionary containing Delta, Gamma, Theta, Vega, and Rho
        """
        if time_to_expiry <= 0:
            return self._zero_greeks()
        
        d1, d2 = self._calculate_d1_d2(spot, strike, time_to_expiry,
                                      risk_free_rate, volatility)
        
        # Common terms
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        
        sqrt_t = math.sqrt(time_to_expiry)
        discount_factor = np.exp(-risk_free_rate * time_to_expiry)
        div_discount = np.exp(-self.dividend_yield * time_to_expiry)
        
        if option_type.lower() == 'call':
            # Call Greeks
            delta = div_discount * cdf_d1
            theta = (-(spot * pdf_d1 * volatility * div_discount) / (2 * sqrt_t) -
                    risk_free_rate * strike * discount_factor * cdf_d2 +
                    self.dividend_yield * spot * div_discount * cdf_d1)
            rho = strike * time_to_expiry * discount_factor * cdf_d2
        else:
            # Put Greeks
            delta = div_discount * (cdf_d1 - 1)
            theta = (-(spot * pdf_d1 * volatility * div_discount) / (2 * sqrt_t) +
                    risk_free_rate * strike * discount_factor * norm.cdf(-d2) -
                    self.dividend_yield * spot * div_discount * norm.cdf(-d1))
            rho = -strike * time_to_expiry * discount_factor * norm.cdf(-d2)
        
        # Greeks that are the same for calls and puts
        gamma = (div_discount * pdf_d1) / (spot * volatility * sqrt_t)
        vega = spot * div_discount * pdf_d1 * sqrt_t
        
        # Convert to per-dollar and per-percentage basis
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Per day
            'vega': vega / 100,    # Per 1% volatility change
            'rho': rho / 100       # Per 1% rate change
        }
    
    def calculate_implied_volatility(self, market_price: float, spot: float, 
                                   strike: float, time_to_expiry: float,
                                   risk_free_rate: float, option_type: str = 'call',
                                   max_iterations: int = 100, tolerance: float = 1e-6) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            market_price: Observed market price
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility or None if not found
        """
        if time_to_expiry <= 0:
            return None
        
        # Initial guess
        volatility = 0.2
        
        for _ in range(max_iterations):
            # Calculate price and vega
            price = self.calculate_price(spot, strike, time_to_expiry,
                                       risk_free_rate, volatility, option_type)
            greeks = self.calculate_greeks(spot, strike, time_to_expiry,
                                         risk_free_rate, volatility, option_type)
            vega = greeks['vega'] * 100  # Convert back to per-unit basis
            
            # Check for convergence
            price_diff = price - market_price
            if abs(price_diff) < tolerance:
                return volatility
            
            # Newton-Raphson update
            if abs(vega) < 1e-10:  # Avoid division by zero
                break
            
            volatility = volatility - price_diff / vega
            
            # Keep volatility positive and reasonable
            volatility = max(0.001, min(5.0, volatility))
        
        return None  # Failed to converge
    
    def _calculate_d1_d2(self, spot: float, strike: float, time_to_expiry: float,
                        risk_free_rate: float, volatility: float) -> tuple:
        """Calculate d1 and d2 parameters"""
        sqrt_t = math.sqrt(time_to_expiry)
        
        d1 = (math.log(spot / strike) + 
              (risk_free_rate - self.dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t
        
        return d1, d2
    
    def _intrinsic_value(self, spot: float, strike: float, option_type: str) -> float:
        """Calculate intrinsic value at expiry"""
        if option_type.lower() == 'call':
            return max(0, spot - strike)
        else:
            return max(0, strike - spot)
    
    def _zero_greeks(self) -> Dict[str, float]:
        """Return zero Greeks for expired options"""
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }


class BinomialModel(OptionsPricingModel):
    """
    Binomial tree model for American options
    
    Can handle American exercise, discrete dividends,
    and provides more flexibility than Black-Scholes
    """
    
    def __init__(self, steps: int = 100, dividend_yield: float = 0.0):
        self.steps = steps
        self.dividend_yield = dividend_yield
    
    def calculate_price(self, spot: float, strike: float, time_to_expiry: float,
                       risk_free_rate: float, volatility: float,
                       option_type: str = 'call', exercise_style: str = 'european') -> float:
        """
        Calculate option price using binomial tree
        
        Args:
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            volatility: Annualized volatility
            option_type: 'call' or 'put'
            exercise_style: 'european' or 'american'
            
        Returns:
            Option price
        """
        if time_to_expiry <= 0:
            return max(0, self._intrinsic_value(spot, strike, option_type))
        
        # Tree parameters
        dt = time_to_expiry / self.steps
        u = math.exp(volatility * math.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (math.exp((risk_free_rate - self.dividend_yield) * dt) - d) / (u - d)  # Risk-neutral probability
        discount = math.exp(-risk_free_rate * dt)
        
        # Initialize asset prices at final nodes
        asset_prices = np.array([spot * (u ** (self.steps - i)) * (d ** i) 
                               for i in range(self.steps + 1)])
        
        # Initialize option values at final nodes
        if option_type.lower() == 'call':
            option_values = np.maximum(asset_prices - strike, 0)
        else:
            option_values = np.maximum(strike - asset_prices, 0)
        
        # Backward induction
        for step in range(self.steps - 1, -1, -1):
            # Calculate asset prices at current step
            current_prices = np.array([spot * (u ** (step - i)) * (d ** i)
                                     for i in range(step + 1)])
            
            # Calculate continuation values
            continuation_values = discount * (p * option_values[:-1] + (1 - p) * option_values[1:])
            
            if exercise_style.lower() == 'american':
                # Calculate intrinsic values
                if option_type.lower() == 'call':
                    intrinsic_values = np.maximum(current_prices - strike, 0)
                else:
                    intrinsic_values = np.maximum(strike - current_prices, 0)
                
                # Take maximum of continuation and intrinsic values
                option_values = np.maximum(continuation_values, intrinsic_values)
            else:
                # European exercise - just use continuation values
                option_values = continuation_values
        
        return option_values[0]
    
    def calculate_greeks(self, spot: float, strike: float, time_to_expiry: float,
                        risk_free_rate: float, volatility: float,
                        option_type: str = 'call', exercise_style: str = 'european') -> Dict[str, float]:
        """
        Calculate Greeks using finite difference approximation
        
        Returns:
            Dictionary containing Delta, Gamma, Theta, Vega, and Rho
        """
        if time_to_expiry <= 0:
            return self._zero_greeks()
        
        # Base price
        base_price = self.calculate_price(spot, strike, time_to_expiry,
                                        risk_free_rate, volatility, 
                                        option_type, exercise_style)
        
        # Delta calculation (finite difference)
        ds = spot * 0.01  # 1% bump
        price_up = self.calculate_price(spot + ds, strike, time_to_expiry,
                                      risk_free_rate, volatility,
                                      option_type, exercise_style)
        price_down = self.calculate_price(spot - ds, strike, time_to_expiry,
                                        risk_free_rate, volatility,
                                        option_type, exercise_style)
        delta = (price_up - price_down) / (2 * ds)
        
        # Gamma calculation
        gamma = (price_up - 2 * base_price + price_down) / (ds ** 2)
        
        # Theta calculation (1 day time decay)
        dt = 1 / 365
        if time_to_expiry > dt:
            price_theta = self.calculate_price(spot, strike, time_to_expiry - dt,
                                             risk_free_rate, volatility,
                                             option_type, exercise_style)
            theta = -(price_theta - base_price)  # Price change per day
        else:
            theta = 0
        
        # Vega calculation (1% volatility bump)
        dv = 0.01
        price_vega = self.calculate_price(spot, strike, time_to_expiry,
                                        risk_free_rate, volatility + dv,
                                        option_type, exercise_style)
        vega = price_vega - base_price
        
        # Rho calculation (1% rate bump)
        dr = 0.01
        price_rho = self.calculate_price(spot, strike, time_to_expiry,
                                       risk_free_rate + dr, volatility,
                                       option_type, exercise_style)
        rho = price_rho - base_price
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def _intrinsic_value(self, spot: float, strike: float, option_type: str) -> float:
        """Calculate intrinsic value"""
        if option_type.lower() == 'call':
            return max(0, spot - strike)
        else:
            return max(0, strike - spot)
    
    def _zero_greeks(self) -> Dict[str, float]:
        """Return zero Greeks for expired options"""
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }


class MonteCarloModel(OptionsPricingModel):
    """
    Monte Carlo simulation for exotic options
    
    Useful for path-dependent options like Asian options,
    barrier options, and other exotic structures
    """
    
    def __init__(self, num_simulations: int = 10000, num_steps: int = 252,
                 dividend_yield: float = 0.0, random_seed: Optional[int] = None):
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.dividend_yield = dividend_yield
        if random_seed:
            np.random.seed(random_seed)
    
    def calculate_price(self, spot: float, strike: float, time_to_expiry: float,
                       risk_free_rate: float, volatility: float,
                       option_type: str = 'call') -> float:
        """
        Calculate European option price using Monte Carlo
        
        This is a basic implementation - can be extended for exotic options
        """
        if time_to_expiry <= 0:
            return max(0, self._intrinsic_value(spot, strike, option_type))
        
        dt = time_to_expiry / self.num_steps
        drift = risk_free_rate - self.dividend_yield - 0.5 * volatility**2
        
        # Generate random paths
        random_increments = np.random.normal(0, 1, (self.num_simulations, self.num_steps))
        
        # Calculate final spot prices
        cumulative_returns = np.cumsum(
            drift * dt + volatility * math.sqrt(dt) * random_increments, axis=1
        )
        final_spots = spot * np.exp(cumulative_returns[:, -1])
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_spots - strike, 0)
        else:
            payoffs = np.maximum(strike - final_spots, 0)
        
        # Discount to present value
        present_value = np.exp(-risk_free_rate * time_to_expiry) * np.mean(payoffs)
        
        return present_value
    
    def calculate_greeks(self, spot: float, strike: float, time_to_expiry: float,
                        risk_free_rate: float, volatility: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate Greeks using finite difference with Monte Carlo
        
        Note: This is computationally expensive and may have higher noise
        """
        if time_to_expiry <= 0:
            return self._zero_greeks()
        
        # Use the same random seed for all calculations to reduce noise
        original_state = np.random.get_state()
        
        # Base price
        np.random.set_state(original_state)
        base_price = self.calculate_price(spot, strike, time_to_expiry,
                                        risk_free_rate, volatility, option_type)
        
        # Delta
        ds = spot * 0.01
        np.random.set_state(original_state)
        price_up = self.calculate_price(spot + ds, strike, time_to_expiry,
                                      risk_free_rate, volatility, option_type)
        np.random.set_state(original_state)
        price_down = self.calculate_price(spot - ds, strike, time_to_expiry,
                                        risk_free_rate, volatility, option_type)
        delta = (price_up - price_down) / (2 * ds)
        
        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (ds ** 2)
        
        # Vega
        dv = 0.01
        np.random.set_state(original_state)
        price_vega = self.calculate_price(spot, strike, time_to_expiry,
                                        risk_free_rate, volatility + dv, option_type)
        vega = price_vega - base_price
        
        # Theta and Rho (simplified)
        theta = 0  # Would need more sophisticated implementation
        rho = 0    # Would need more sophisticated implementation
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def _intrinsic_value(self, spot: float, strike: float, option_type: str) -> float:
        """Calculate intrinsic value"""
        if option_type.lower() == 'call':
            return max(0, spot - strike)
        else:
            return max(0, strike - spot)
    
    def _zero_greeks(self) -> Dict[str, float]:
        """Return zero Greeks for expired options"""
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
