"""
Example script demonstrating how to use the Aave position fetcher.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qcollat_simplifier import AavePositionFetcher
from qcollat_simplifier.config import DEFAULT_BATCH_SIZE

def main():
    """Example usage of the Aave position fetcher."""
    
    print("Aave v3 Position Fetcher Example")
    print("=" * 40)
    
    # Initialize the fetcher
    print("Initializing fetcher...")
    fetcher = AavePositionFetcher(
        rate_limit_delay=0.3,  # Slightly slower for reliability
        max_retries=3
    )
    
    # Fetch a small sample for demonstration
    print("Fetching sample positions...")
    df = fetcher.fetch_positions(
        batch_size=DEFAULT_BATCH_SIZE,
        max_positions=100  # Small sample for demo
    )
    
    if df.empty:
        print("No positions found!")
        return
    
    # Display summary
    print(f"\nFetched {len(df)} positions")
    print(f"Unique users: {df['user'].nunique()}")
    print(f"Unique tokens: {df['reserveSymbol'].nunique()}")
    print(f"Total collateral (ETH): {df['collateralBalanceETH'].sum():.2f}")
    print(f"Total debt (ETH): {df['debtBalanceETH'].sum():.2f}")
    
    # Display sample data
    print("\nSample positions:")
    print(df.head(10))
    
    # Save to CSV
    print("\nSaving to CSV...")
    filepath = fetcher.save_positions(df, "example_positions.csv")
    print(f"Saved to: {filepath}")
    
    # Show some statistics
    print("\nPosition Statistics:")
    print(f"Average collateral per position: {df['collateralBalanceETH'].mean():.4f} ETH")
    print(f"Average debt per position: {df['debtBalanceETH'].mean():.4f} ETH")
    print(f"Positions with debt: {(df['debtBalanceETH'] > 0).sum()}")
    print(f"Positions with collateral only: {(df['debtBalanceETH'] == 0).sum()}")
    
    # Top tokens by collateral
    print("\nTop tokens by total collateral:")
    token_collateral = df.groupby('reserveSymbol')['collateralBalanceETH'].sum().sort_values(ascending=False)
    print(token_collateral.head())

if __name__ == "__main__":
    main() 