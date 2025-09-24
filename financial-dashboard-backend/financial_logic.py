"""
Financial Logic Module

This module provides advanced financial calculation functions using Pandas and NumPy.
Designed for integration with FastAPI backends and financial dashboard applications.

Functions:
    - calculate_net_worth_timeline: Calculates cumulative net worth over time
    - calculate_asset_allocation: Determines portfolio allocation percentages
    - run_monte_carlo_simulation: Performs Monte Carlo analysis for investment projections

Author: Financial Dashboard Backend
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_net_worth_timeline(transactions: List[Dict]) -> pd.DataFrame:
    """
    Calculate cumulative net worth over time based on transaction history.
    
    This function takes a list of financial transactions and computes a running
    total to show how net worth changes over time. Positive amounts represent
    income/gains, negative amounts represent expenses/losses.
    
    Args:
        transactions (List[Dict]): List of transaction dictionaries containing:
            - 'date': Transaction date (string or datetime object)
            - 'amount': Transaction amount (float, positive for income, negative for expenses)
            - Other fields are ignored for this calculation
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'date': Transaction dates (datetime)
            - 'cumulative_net_worth': Running total of net worth
    
    Raises:
        ValueError: If transactions list is empty or missing required fields
        
    Example:
        >>> transactions = [
        ...     {'date': '2024-01-01', 'amount': 1000.0, 'category': 'Salary'},
        ...     {'date': '2024-01-15', 'amount': -200.0, 'category': 'Groceries'}
        ... ]
        >>> df = calculate_net_worth_timeline(transactions)
        >>> print(df)
                date  cumulative_net_worth
        0 2024-01-01                1000.0
        1 2024-01-15                 800.0
    """
    try:
        # Input validation
        if not transactions:
            logger.warning("Empty transactions list provided")
            return pd.DataFrame(columns=['date', 'cumulative_net_worth'])
        
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(transactions)
        
        # Validate required columns
        if 'date' not in df.columns or 'amount' not in df.columns:
            raise ValueError("Transactions must contain 'date' and 'amount' fields")
        
        # Convert date column to datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date in ascending order to ensure proper chronological ordering
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate cumulative sum of amounts to get running net worth
        df['cumulative_net_worth'] = df['amount'].cumsum()
        
        # Create result DataFrame with only required columns
        result_df = df[['date', 'cumulative_net_worth']].copy()
        
        logger.info(f"Calculated net worth timeline for {len(result_df)} transactions")
        logger.info(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")
        logger.info(f"Final net worth: ${result_df['cumulative_net_worth'].iloc[-1]:,.2f}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error calculating net worth timeline: {str(e)}")
        raise


def calculate_asset_allocation(portfolio_holdings: List[Dict], 
                             current_prices: Dict[str, float]) -> pd.DataFrame:
    """
    Calculate asset allocation percentages for a portfolio based on current market prices.
    
    This function determines what percentage of the total portfolio value each holding
    represents, providing insights into portfolio diversification and risk exposure.
    
    Args:
        portfolio_holdings (List[Dict]): List of holding dictionaries containing:
            - 'ticker': Stock ticker symbol (string)
            - 'shares': Number of shares owned (float)
        current_prices (Dict[str, float]): Dictionary mapping ticker symbols to current prices
            Example: {'AAPL': 180.50, 'GOOGL': 2800.00}
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'ticker': Stock ticker symbols
            - 'shares': Number of shares owned
            - 'current_price': Current price per share
            - 'total_value': Total value of holding (shares * price)
            - 'percentage': Percentage of total portfolio value
    
    Raises:
        ValueError: If holdings list is empty or price data is missing
        
    Example:
        >>> holdings = [
        ...     {'ticker': 'AAPL', 'shares': 10.0},
        ...     {'ticker': 'GOOGL', 'shares': 5.0}
        ... ]
        >>> prices = {'AAPL': 180.50, 'GOOGL': 2800.00}
        >>> df = calculate_asset_allocation(holdings, prices)
        >>> print(df)
          ticker  shares  current_price  total_value  percentage
        0   AAPL    10.0         180.50       1805.0       11.41
        1  GOOGL     5.0        2800.00      14000.0       88.59
    """
    try:
        # Input validation
        if not portfolio_holdings:
            logger.warning("Empty portfolio holdings provided")
            return pd.DataFrame(columns=['ticker', 'shares', 'current_price', 'total_value', 'percentage'])
        
        if not current_prices:
            raise ValueError("Current prices dictionary cannot be empty")
        
        # Convert holdings list to DataFrame
        df = pd.DataFrame(portfolio_holdings)
        
        # Validate required columns
        if 'ticker' not in df.columns or 'shares' not in df.columns:
            raise ValueError("Holdings must contain 'ticker' and 'shares' fields")
        
        # Add current price for each holding
        df['current_price'] = df['ticker'].map(current_prices)
        
        # Check for missing price data
        missing_prices = df[df['current_price'].isna()]['ticker'].tolist()
        if missing_prices:
            logger.warning(f"Missing price data for tickers: {missing_prices}")
            # Remove holdings without price data
            df = df.dropna(subset=['current_price'])
        
        if df.empty:
            logger.warning("No holdings with valid price data")
            return pd.DataFrame(columns=['ticker', 'shares', 'current_price', 'total_value', 'percentage'])
        
        # Calculate total value for each holding (shares * current price)
        df['total_value'] = df['shares'] * df['current_price']
        
        # Calculate total portfolio value
        total_portfolio_value = df['total_value'].sum()
        
        if total_portfolio_value == 0:
            raise ValueError("Total portfolio value cannot be zero")
        
        # Calculate percentage allocation for each holding
        df['percentage'] = (df['total_value'] / total_portfolio_value * 100).round(2)
        
        # Sort by percentage in descending order (largest holdings first)
        df = df.sort_values('percentage', ascending=False).reset_index(drop=True)
        
        logger.info(f"Calculated asset allocation for {len(df)} holdings")
        logger.info(f"Total portfolio value: ${total_portfolio_value:,.2f}")
        logger.info(f"Largest holding: {df.iloc[0]['ticker']} ({df.iloc[0]['percentage']:.2f}%)")
        
        return df[['ticker', 'shares', 'current_price', 'total_value', 'percentage']]
        
    except Exception as e:
        logger.error(f"Error calculating asset allocation: {str(e)}")
        raise


def run_monte_carlo_simulation(initial_investment: float,
                             time_period: int,
                             expected_return: float,
                             volatility: float,
                             num_simulations: int = 1000) -> pd.DataFrame:
    """
    Run Monte Carlo simulation for investment portfolio projections.
    
    This function uses random walk modeling to simulate potential future portfolio
    values based on expected returns and volatility. It helps assess investment
    risk and potential outcomes over a specified time period.
    
    Args:
        initial_investment (float): Starting investment amount in dollars
        time_period (int): Investment time period in years
        expected_return (float): Expected annual return as decimal (e.g., 0.08 for 8%)
        volatility (float): Annual volatility as decimal (e.g., 0.15 for 15%)
        num_simulations (int, optional): Number of simulation paths to generate. Defaults to 1000.
    
    Returns:
        pd.DataFrame: DataFrame where:
            - Index represents time steps (0 to time_period years)
            - Each column represents one simulation path
            - Values represent projected portfolio value at each time step
            
    Raises:
        ValueError: If input parameters are invalid
        
    Example:
        >>> df = run_monte_carlo_simulation(
        ...     initial_investment=10000.0,
        ...     time_period=10,
        ...     expected_return=0.08,
        ...     volatility=0.15,
        ...     num_simulations=1000
        ... )
        >>> print(f"Simulated final values range: ${df.iloc[-1].min():,.0f} - ${df.iloc[-1].max():,.0f}")
        >>> print(f"Mean final value: ${df.iloc[-1].mean():,.0f}")
    """
    try:
        # Input validation
        if initial_investment <= 0:
            raise ValueError("Initial investment must be positive")
        if time_period <= 0:
            raise ValueError("Time period must be positive")
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        if volatility < 0:
            raise ValueError("Volatility cannot be negative")
        
        logger.info(f"Starting Monte Carlo simulation with {num_simulations} paths over {time_period} years")
        logger.info(f"Parameters: Initial=${initial_investment:,.0f}, Return={expected_return:.1%}, Volatility={volatility:.1%}")
        
        # Set random seed for reproducible results (optional - remove for true randomness)
        np.random.seed(42)
        
        # Number of time steps (assuming monthly steps for more granular simulation)
        time_steps = time_period * 12  # Monthly time steps
        dt = 1/12  # Time increment (1 month as fraction of year)
        
        # Initialize array to store all simulation paths
        # Shape: (time_steps + 1, num_simulations) - +1 to include initial value
        simulation_results = np.zeros((time_steps + 1, num_simulations))
        
        # Set initial investment value for all simulations
        simulation_results[0, :] = initial_investment
        
        # Generate random returns for each time step and simulation
        # Using geometric Brownian motion formula: dS/S = μdt + σdW
        for t in range(1, time_steps + 1):
            # Generate random normal variables (Brownian motion increments)
            random_shocks = np.random.normal(0, 1, num_simulations)
            
            # Calculate returns using geometric Brownian motion
            # Drift component: (expected_return - 0.5 * volatility^2) * dt
            # Diffusion component: volatility * sqrt(dt) * random_shock
            drift = (expected_return - 0.5 * volatility**2) * dt
            diffusion = volatility * np.sqrt(dt) * random_shocks
            
            # Apply returns to get new portfolio values
            simulation_results[t, :] = simulation_results[t-1, :] * np.exp(drift + diffusion)
        
        # Convert to DataFrame with meaningful column names
        column_names = [f'Simulation_{i+1}' for i in range(num_simulations)]
        
        # Create time index (in years)
        time_index = np.linspace(0, time_period, time_steps + 1)
        
        df = pd.DataFrame(
            simulation_results,
            columns=column_names,
            index=time_index
        )
        
        # Add summary statistics
        final_values = df.iloc[-1]
        
        logger.info("Monte Carlo simulation completed successfully")
        logger.info(f"Final value statistics:")
        logger.info(f"  Mean: ${final_values.mean():,.0f}")
        logger.info(f"  Median: ${final_values.median():,.0f}")
        logger.info(f"  Min: ${final_values.min():,.0f}")
        logger.info(f"  Max: ${final_values.max():,.0f}")
        logger.info(f"  Standard deviation: ${final_values.std():,.0f}")
        
        # Calculate probability of positive returns
        positive_return_prob = (final_values > initial_investment).mean() * 100
        logger.info(f"Probability of positive returns: {positive_return_prob:.1f}%")
        
        return df
        
    except Exception as e:
        logger.error(f"Error running Monte Carlo simulation: {str(e)}")
        raise


def get_simulation_summary(simulation_df: pd.DataFrame) -> Dict[str, Union[float, int]]:
    """
    Calculate summary statistics for Monte Carlo simulation results.
    
    Args:
        simulation_df (pd.DataFrame): DataFrame from run_monte_carlo_simulation()
        
    Returns:
        Dict containing summary statistics
    """
    final_values = simulation_df.iloc[-1]
    initial_value = simulation_df.iloc[0].iloc[0]  # All simulations start with same value
    
    return {
        'initial_value': initial_value,
        'mean_final_value': final_values.mean(),
        'median_final_value': final_values.median(),
        'min_final_value': final_values.min(),
        'max_final_value': final_values.max(),
        'std_final_value': final_values.std(),
        'probability_positive_return': (final_values > initial_value).mean() * 100,
        'percentile_5': final_values.quantile(0.05),
        'percentile_95': final_values.quantile(0.95),
        'num_simulations': len(final_values)
    }


# Example usage and testing functions
if __name__ == "__main__":
    """
    Example usage of all financial logic functions.
    Run this file directly to see demonstrations of each function.
    """
    print("=" * 60)
    print("FINANCIAL LOGIC MODULE - DEMONSTRATION")
    print("=" * 60)
    
    # Example 1: Net Worth Timeline
    print("\n1. NET WORTH TIMELINE CALCULATION")
    print("-" * 40)
    
    sample_transactions = [
        {'date': '2024-01-01', 'amount': 5000.0, 'category': 'Salary', 'description': 'January salary'},
        {'date': '2024-01-05', 'amount': -1200.0, 'category': 'Rent', 'description': 'Monthly rent'},
        {'date': '2024-01-10', 'amount': -300.0, 'category': 'Food', 'description': 'Groceries'},
        {'date': '2024-01-15', 'amount': 2000.0, 'category': 'Investment', 'description': 'Stock dividend'},
        {'date': '2024-01-20', 'amount': -500.0, 'category': 'Utilities', 'description': 'Electric bill'},
    ]
    
    net_worth_df = calculate_net_worth_timeline(sample_transactions)
    print(net_worth_df.to_string(index=False))
    
    # Example 2: Asset Allocation
    print("\n\n2. ASSET ALLOCATION CALCULATION")
    print("-" * 40)
    
    sample_holdings = [
        {'ticker': 'AAPL', 'shares': 50.0},
        {'ticker': 'GOOGL', 'shares': 10.0},
        {'ticker': 'MSFT', 'shares': 25.0},
        {'ticker': 'TSLA', 'shares': 15.0},
    ]
    
    sample_prices = {
        'AAPL': 180.50,
        'GOOGL': 2800.00,
        'MSFT': 380.75,
        'TSLA': 250.30
    }
    
    allocation_df = calculate_asset_allocation(sample_holdings, sample_prices)
    print(allocation_df.to_string(index=False))
    
    # Example 3: Monte Carlo Simulation
    print("\n\n3. MONTE CARLO SIMULATION")
    print("-" * 40)
    
    simulation_df = run_monte_carlo_simulation(
        initial_investment=100000.0,
        time_period=10,
        expected_return=0.08,
        volatility=0.15,
        num_simulations=1000
    )
    
    # Show summary statistics
    summary = get_simulation_summary(simulation_df)
    print(f"Initial Investment: ${summary['initial_value']:,.0f}")
    print(f"Mean Final Value: ${summary['mean_final_value']:,.0f}")
    print(f"Median Final Value: ${summary['median_final_value']:,.0f}")
    print(f"5th Percentile: ${summary['percentile_5']:,.0f}")
    print(f"95th Percentile: ${summary['percentile_95']:,.0f}")
    print(f"Probability of Positive Return: {summary['probability_positive_return']:.1f}%")
    
    print(f"\nSimulation DataFrame shape: {simulation_df.shape}")
    print("First few rows of simulation data:")
    print(simulation_df.iloc[:5, :3].round(2))  # Show first 5 time steps, first 3 simulations
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)