"""
Build bipartite graph from Aave position data.

This script loads the latest CSV file with Aave positions and constructs
a bipartite graph using NetworkX, then exports the adjacency matrix
as a SciPy CSR sparse matrix.
"""

import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import logging
import os
import glob
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AaveGraphBuilder:
    """Builds bipartite graphs from Aave position data."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "graphs"):
        """
        Initialize the graph builder.
        
        Args:
            data_dir: Directory containing CSV files
            output_dir: Directory to save graph outputs
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Graph components
        self.G = None
        self.asset_nodes = []
        self.position_nodes = []
        self.node_mapping = {}
        self.reverse_mapping = {}
        
    def find_latest_csv(self) -> Optional[Path]:
        """
        Find the latest CSV file in the data directory.
        
        Returns:
            Path to the latest CSV file or None if not found
        """
        pattern = self.data_dir / "aave_positions_*.csv"
        csv_files = list(glob.glob(str(pattern)))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.data_dir}")
            return None
        
        # Sort by modification time (latest first)
        csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_file = Path(csv_files[0])
        
        logger.info(f"Found latest CSV file: {latest_file}")
        return latest_file
    
    def load_data(self, csv_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load position data from CSV file.
        
        Args:
            csv_path: Path to CSV file (if None, uses latest)
            
        Returns:
            DataFrame with position data
        """
        if csv_path is None:
            csv_path = self.find_latest_csv()
            
        if csv_path is None:
            raise FileNotFoundError("No CSV file found")
        
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['user', 'reserveSymbol', 'balance', 'principal', 'side', 'type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        df = df.dropna(subset=['reserveSymbol', 'user'])
        df = df[(df['balance'] > 0) | (df['principal'] > 0)]
        
        # Create synthetic columns to match expected structure
        df['txHash'] = df['hashOpened'].fillna('unknown')
        df['collateralBalanceETH'] = df.apply(
            lambda row: row['balance'] if row['side'] == 'COLLATERAL' else 0.0, axis=1
        )
        df['debtBalanceETH'] = df.apply(
            lambda row: row['principal'] if row['side'] == 'BORROWER' else 0.0, axis=1
        )
        
        logger.info(f"Loaded {len(df)} positions")
        logger.info(f"Unique assets: {df['reserveSymbol'].nunique()}")
        logger.info(f"Unique users: {df['user'].nunique()}")
        logger.info(f"Collateral positions: {len(df[df['side'] == 'COLLATERAL'])}")
        logger.info(f"Borrower positions: {len(df[df['side'] == 'BORROWER'])}")
        
        return df
    
    def create_position_id(self, row: pd.Series) -> str:
        """
        Create a unique position identifier.
        
        Args:
            row: DataFrame row with position data
            
        Returns:
            Unique position identifier
        """
        # Use user-reserve pair as position identifier
        return f"{row['user']}_{row['reserveSymbol']}"
    
    def normalize_weights(self, values: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize edge weights.
        
        Args:
            values: Array of values to normalize
            method: Normalization method ('minmax', 'zscore', 'log', 'none')
            
        Returns:
            Normalized values
        """
        if method == 'minmax':
            min_val = values.min()
            max_val = values.max()
            if max_val > min_val:
                return (values - min_val) / (max_val - min_val)
            else:
                return np.ones_like(values)
        elif method == 'zscore':
            mean_val = values.mean()
            std_val = values.std()
            if std_val > 0:
                return (values - mean_val) / std_val
            else:
                return np.zeros_like(values)
        elif method == 'log':
            # Add small constant to avoid log(0)
            return np.log1p(values)
        elif method == 'none':
            return values
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def build_bipartite_graph(self, df: pd.DataFrame, 
                            weight_type: str = 'collateral',
                            normalization: str = 'minmax',
                            min_weight: float = 0.0) -> nx.Graph:
        """
        Build bipartite graph from position data.
        
        Args:
            df: DataFrame with position data
            weight_type: Type of weight ('collateral', 'debt', 'combined')
            normalization: Normalization method for weights
            min_weight: Minimum weight threshold
            
        Returns:
            NetworkX bipartite graph
        """
        logger.info(f"Building bipartite graph with weight_type={weight_type}, normalization={normalization}")
        
        # Create graph
        G = nx.Graph()
        
        # Get unique assets and positions
        assets = sorted(df['reserveSymbol'].unique())
        positions = []
        
        # Create position identifiers
        df['position_id'] = df.apply(self.create_position_id, axis=1)
        positions = sorted(df['position_id'].unique())
        
        # Store node lists
        self.asset_nodes = assets
        self.position_nodes = positions
        
        # Create node mapping for matrix construction
        self.node_mapping = {}
        self.reverse_mapping = {}
        
        # Map assets to indices 0 to len(assets)-1
        for i, asset in enumerate(assets):
            self.node_mapping[asset] = i
            self.reverse_mapping[i] = asset
        
        # Map positions to indices len(assets) to len(assets)+len(positions)-1
        for i, position in enumerate(positions):
            self.node_mapping[position] = len(assets) + i
            self.reverse_mapping[len(assets) + i] = position
        
        # Add nodes
        G.add_nodes_from(assets, bipartite=0)  # Asset nodes
        G.add_nodes_from(positions, bipartite=1)  # Position nodes
        
        # Add edges
        edge_count = 0
        for _, row in df.iterrows():
            asset = row['reserveSymbol']
            position = row['position_id']
            
            # Calculate weight based on type
            if weight_type == 'collateral':
                weight = row['collateralBalanceETH']
            elif weight_type == 'debt':
                weight = row['debtBalanceETH']
            elif weight_type == 'combined':
                weight = row['collateralBalanceETH'] + row['debtBalanceETH']
            else:
                raise ValueError(f"Unknown weight_type: {weight_type}")
            
            # Skip if weight is below threshold
            if weight <= min_weight:
                continue
            
            # Add edge
            G.add_edge(asset, position, weight=weight)
            edge_count += 1
        
        logger.info(f"Added {edge_count} edges to graph")
        
        # Normalize edge weights
        if normalization != 'none':
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            normalized_weights = self.normalize_weights(np.array(weights), normalization)
            
            for i, (u, v) in enumerate(G.edges()):
                G[u][v]['weight'] = normalized_weights[i]
            
            logger.info(f"Normalized edge weights using {normalization} method")
        
        self.G = G
        return G
    
    def create_adjacency_matrix(self) -> sp.csr_matrix:
        """
        Create adjacency matrix from the graph.
        
        Returns:
            SciPy CSR sparse matrix
        """
        if self.G is None:
            raise ValueError("Graph not built yet. Call build_bipartite_graph() first.")
        
        logger.info("Creating adjacency matrix...")
        
        # Get node order for matrix
        all_nodes = self.asset_nodes + self.position_nodes
        n_nodes = len(all_nodes)
        
        # Create adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.G, nodelist=all_nodes, weight='weight')
        
        # Convert to CSR format
        adj_matrix_csr = sp.csr_matrix(adj_matrix)
        
        logger.info(f"Created adjacency matrix: {adj_matrix_csr.shape}")
        logger.info(f"Matrix density: {adj_matrix_csr.nnz / (adj_matrix_csr.shape[0] * adj_matrix_csr.shape[1]):.6f}")
        
        return adj_matrix_csr
    
    def save_graph(self, adj_matrix: sp.csr_matrix, 
                  filename: Optional[str] = None,
                  save_metadata: bool = True) -> str:
        """
        Save graph data to files.
        
        Args:
            adj_matrix: Adjacency matrix to save
            filename: Base filename (without extension)
            save_metadata: Whether to save node metadata
            
        Returns:
            Path to saved adjacency matrix file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"A_{timestamp}"
        
        # Save adjacency matrix
        matrix_path = self.output_dir / f"{filename}.npz"
        sp.save_npz(matrix_path, adj_matrix)
        logger.info(f"Saved adjacency matrix to {matrix_path}")
        
        # Save metadata
        if save_metadata:
            metadata_path = self.output_dir / f"{filename}_metadata.npz"
            np.savez_compressed(
                metadata_path,
                asset_nodes=np.array(self.asset_nodes),
                position_nodes=np.array(self.position_nodes),
                node_mapping=np.array(list(self.node_mapping.items()), dtype=object)
            )
            logger.info(f"Saved metadata to {metadata_path}")
        
        return str(matrix_path)
    
    def load_graph(self, matrix_path: str, metadata_path: Optional[str] = None) -> Tuple[sp.csr_matrix, Dict]:
        """
        Load graph data from files.
        
        Args:
            matrix_path: Path to adjacency matrix file
            metadata_path: Path to metadata file (optional)
            
        Returns:
            Tuple of (adjacency_matrix, metadata_dict)
        """
        # Load adjacency matrix
        adj_matrix = sp.load_npz(matrix_path)
        logger.info(f"Loaded adjacency matrix: {adj_matrix.shape}")
        
        metadata = {}
        
        # Load metadata if provided
        if metadata_path and os.path.exists(metadata_path):
            metadata_data = np.load(metadata_path, allow_pickle=True)
            metadata = {
                'asset_nodes': metadata_data['asset_nodes'].tolist(),
                'position_nodes': metadata_data['position_nodes'].tolist(),
                'node_mapping': dict(metadata_data['node_mapping'].tolist())
            }
            logger.info("Loaded metadata")
        
        return adj_matrix, metadata
    
    def get_graph_statistics(self) -> Dict:
        """
        Get statistics about the built graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if self.G is None:
            raise ValueError("Graph not built yet. Call build_bipartite_graph() first.")
        
        stats = {
            'num_assets': len(self.asset_nodes),
            'num_positions': len(self.position_nodes),
            'num_edges': self.G.number_of_edges(),
            'is_bipartite': nx.is_bipartite(self.G),
            'density': nx.density(self.G),
            'avg_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),
        }
        
        # Edge weight statistics
        weights = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        if weights:
            stats.update({
                'min_weight': min(weights),
                'max_weight': max(weights),
                'avg_weight': np.mean(weights),
                'std_weight': np.std(weights)
            })
        
        return stats


def main():
    """Main function to build and save graph."""
    try:
        # Initialize builder
        builder = AaveGraphBuilder()
        
        # Load data
        df = builder.load_data()
        
        # Build graph with collateral weights
        G = builder.build_bipartite_graph(
            df, 
            weight_type='collateral',
            normalization='minmax',
            min_weight=0.001  # Minimum 0.001 ETH
        )
        
        # Create adjacency matrix
        adj_matrix = builder.create_adjacency_matrix()
        
        # Get statistics
        stats = builder.get_graph_statistics()
        logger.info("Graph statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Save graph
        matrix_path = builder.save_graph(adj_matrix)
        
        logger.info(f"Successfully built and saved graph to {matrix_path}")
        
        # Display sample of the matrix
        print(f"\nAdjacency matrix shape: {adj_matrix.shape}")
        print(f"Non-zero elements: {adj_matrix.nnz}")
        print(f"Sample of first 5x5 elements:")
        print(adj_matrix[:5, :5].toarray())
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main() 