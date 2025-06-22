"""
Stress test partitions with Monte-Carlo price shocks.

1. Loads labels.npy
2. For each partition, runs Monte-Carlo price-shock on assets inside
3. Recalculates health factors
4. Outputs partition_risk.csv with clusterId, atRiskUSD
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import scipy.stats as stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StressTester:
    """Stress test partitions with Monte-Carlo price shocks."""
    
    def __init__(self, shock_std: float = 0.30, correlation: float = 0.8, n_simulations: int = 1000):
        """
        Initialize stress tester.
        
        Args:
            shock_std: Standard deviation of price shocks (default: 30%)
            correlation: Intra-cluster correlation (default: 0.8)
            n_simulations: Number of Monte-Carlo simulations
        """
        self.shock_std = shock_std
        self.correlation = correlation
        self.n_simulations = n_simulations
        
    def load_data(self, labels_path: str, positions_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load partition labels and position data.
        
        Args:
            labels_path: Path to labels.npy
            positions_path: Path to positions CSV
            
        Returns:
            Tuple of (labels, positions_df)
        """
        logger.info(f"Loading labels from {labels_path}")
        labels = np.load(labels_path)
        
        logger.info(f"Loading positions from {positions_path}")
        positions_df = pd.read_csv(positions_path)
        
        return labels, positions_df
    
    def generate_correlated_shocks(self, n_assets: int) -> np.ndarray:
        """
        Generate correlated price shocks for assets in a cluster.
        
        Args:
            n_assets: Number of assets in the cluster
            
        Returns:
            Array of correlated price shocks
        """
        # Create correlation matrix
        corr_matrix = np.full((n_assets, n_assets), self.correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Generate correlated normal shocks
        shocks = stats.multivariate_normal.rvs(
            mean=np.zeros(n_assets),
            cov=corr_matrix * (self.shock_std ** 2),
            size=self.n_simulations
        )
        
        return shocks
    
    def calculate_health_factor(self, collateral: float, debt: float, 
                              collateral_shock: float, debt_shock: float) -> float:
        """
        Calculate health factor after price shocks.
        
        Args:
            collateral: Original collateral value
            debt: Original debt value
            collateral_shock: Price shock multiplier for collateral
            debt_shock: Price shock multiplier for debt
            
        Returns:
            Health factor (collateral/debt ratio)
        """
        new_collateral = collateral * (1 + collateral_shock)
        new_debt = debt * (1 + debt_shock)
        
        if new_debt <= 0:
            return float('inf')
        
        return new_collateral / new_debt
    
    def stress_test_cluster(self, cluster_positions: pd.DataFrame, 
                          cluster_id: int) -> Dict:
        """
        Stress test a single cluster.
        
        Args:
            cluster_positions: Positions in this cluster
            cluster_id: Cluster identifier
            
        Returns:
            Dictionary with risk metrics
        """
        n_assets = len(cluster_positions)
        if n_assets == 0:
            return {'clusterId': cluster_id, 'atRiskUSD': 0.0}
        
        # Generate correlated shocks for this cluster
        shocks = self.generate_correlated_shocks(n_assets)
        
        # Track positions at risk
        at_risk_count = 0
        total_collateral = 0.0
        
        for _, position in cluster_positions.iterrows():
            collateral = position['collateralBalanceETH']
            debt = position['debtBalanceETH']
            
            if debt <= 0:  # No debt, not at risk
                continue
                
            total_collateral += collateral
            
            # Check each simulation
            for sim in range(self.n_simulations):
                # Use the same shock for both collateral and debt (simplified)
                shock = shocks[sim, 0]  # Use first asset's shock for simplicity
                
                health_factor = self.calculate_health_factor(
                    collateral, debt, shock, shock
                )
                
                if health_factor < 1.0:  # Position is at risk
                    at_risk_count += 1
                    break
        
        # Calculate percentage at risk
        at_risk_pct = at_risk_count / self.n_simulations if self.n_simulations > 0 else 0.0
        at_risk_usd = total_collateral * at_risk_pct
        
        return {
            'clusterId': cluster_id,
            'atRiskUSD': at_risk_usd,
            'totalCollateral': total_collateral,
            'atRiskPercentage': at_risk_pct * 100
        }
    
    def run_stress_test(self, labels: np.ndarray, positions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run stress test on all clusters.
        
        Args:
            labels: Cluster labels for each position
            positions_df: Position data
            
        Returns:
            DataFrame with stress test results
        """
        logger.info("Starting stress test...")
        
        results = []
        unique_clusters = np.unique(labels)
        
        for cluster_id in unique_clusters:
            logger.info(f"Stress testing cluster {cluster_id}")
            
            # Get positions in this cluster
            cluster_mask = labels == cluster_id
            cluster_positions = positions_df[cluster_mask]
            
            # Run stress test
            cluster_result = self.stress_test_cluster(cluster_positions, cluster_id)
            results.append(cluster_result)
        
        return pd.DataFrame(results)
    
    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """
        Save stress test results to CSV.
        
        Args:
            results_df: Results DataFrame
            output_path: Output file path
        """
        # Select only required columns
        output_df = results_df[['clusterId', 'atRiskUSD']].copy()
        output_df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        
        # Also save detailed results
        detailed_path = output_path.replace('.csv', '_detailed.csv')
        results_df.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed results to {detailed_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Stress test partitions with Monte-Carlo shocks")
    parser.add_argument('--labels', type=str, default='graphs/labels.npy', 
                       help='Path to labels.npy')
    parser.add_argument('--positions', type=str, default='data/raw_positions_*.csv',
                       help='Path to positions CSV (supports glob pattern)')
    parser.add_argument('--output', type=str, default='partition_risk.csv',
                       help='Output CSV path')
    parser.add_argument('--shock-std', type=float, default=0.30,
                       help='Price shock standard deviation (default: 0.30)')
    parser.add_argument('--correlation', type=float, default=0.8,
                       help='Intra-cluster correlation (default: 0.8)')
    parser.add_argument('--n-simulations', type=int, default=1000,
                       help='Number of Monte-Carlo simulations (default: 1000)')
    
    args = parser.parse_args()
    
    # Find latest positions file if glob pattern
    if '*' in args.positions:
        import glob
        position_files = glob.glob(args.positions)
        if not position_files:
            raise FileNotFoundError(f"No files found matching pattern: {args.positions}")
        args.positions = max(position_files, key=lambda x: Path(x).stat().st_mtime)
        logger.info(f"Using latest positions file: {args.positions}")
    
    # Initialize stress tester
    stress_tester = StressTester(
        shock_std=args.shock_std,
        correlation=args.correlation,
        n_simulations=args.n_simulations
    )
    
    # Load data
    labels, positions_df = stress_tester.load_data(args.labels, args.positions)
    
    # Run stress test
    results = stress_tester.run_stress_test(labels, positions_df)
    
    # Save results
    stress_tester.save_results(results, args.output)
    
    # Print summary
    print("\nStress Test Summary:")
    print("=" * 50)
    print(f"Total clusters: {len(results)}")
    print(f"Total at-risk USD: ${results['atRiskUSD'].sum():,.2f}")
    print(f"Average at-risk USD per cluster: ${results['atRiskUSD'].mean():,.2f}")
    print(f"Max at-risk USD: ${results['atRiskUSD'].max():,.2f}")
    
    # Show top risky clusters
    print("\nTop 5 Riskiest Clusters:")
    print("-" * 30)
    top_risky = results.nlargest(5, 'atRiskUSD')
    for _, row in top_risky.iterrows():
        print(f"Cluster {row['clusterId']}: ${row['atRiskUSD']:,.2f}")


if __name__ == "__main__":
    main() 