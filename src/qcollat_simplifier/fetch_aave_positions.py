"""
Fetch Aave v3 positions from Polygon subgraph.

This script queries the Aave v3 Polygon subgraph to retrieve user positions
and saves them to a CSV file with proper pagination and rate limiting.
"""

import pandas as pd
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .config import AAVE_V3_POLYGON_SUBGRAPH, GRAPH_API_KEY
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AavePositionFetcher:
    """Fetches Aave v3 positions from Polygon subgraph with pagination and rate limiting."""
    
    def __init__(self, rate_limit_delay: float = 0.1, max_retries: int = 3):
        """
        Initialize the Aave position fetcher.
        
        Args:
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        
        # Aave v3 Polygon subgraph endpoint
        self.subgraph_url = AAVE_V3_POLYGON_SUBGRAPH
        
        # Prepare headers
        self.headers = {
            'User-Agent': 'qcollat-simplifier/0.1.0'
        }
        
        # Add authorization header if API key is provided
        if GRAPH_API_KEY:
            self.headers['Authorization'] = f'Bearer {GRAPH_API_KEY}'
        
        # GraphQL query for user positions
        self.position_query = gql("""
            query GetPositions($first: Int!, $skip: Int!) {
                positions(first: $first, skip: $skip) {
                    id
                    account { id }
                    market { id }
                    asset { id symbol }
                    hashOpened
                    hashClosed
                    balance
                    principal
                    isCollateral
                    isIsolated
                    side
                    type
                }
            }
        """)
        
        # GraphQL query for transaction hashes
        self.tx_query = gql("""
            query GetUserTransactions($user: String!, $first: Int!, $skip: Int!) {
                userTransactions(
                    first: $first
                    skip: $skip
                    where: { user: $user }
                    orderBy: timestamp
                    orderDirection: desc
                ) {
                    id
                    txHash
                    timestamp
                }
            }
        """)
    
    def _create_client(self):
        """Create a new GraphQL client for each request."""
        transport = RequestsHTTPTransport(
            url=self.subgraph_url,
            headers=self.headers
        )
        return Client(transport=transport, fetch_schema_from_transport=False)
    
    def _convert_to_eth(self, amount: str, decimals: int, price_in_eth: str) -> float:
        """
        Convert token amount to ETH equivalent.
        
        Args:
            amount: Token amount as string
            decimals: Token decimals
            price_in_eth: Token price in ETH as string
            
        Returns:
            Amount in ETH equivalent
        """
        try:
            amount_float = float(amount) / (10 ** decimals)
            price_float = float(price_in_eth) / (10 ** 18)  # Price is in wei
            return amount_float * price_float
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _get_latest_tx_hash(self, user_id: str) -> Optional[str]:
        """
        Get the latest transaction hash for a user.
        
        Args:
            user_id: User address
            
        Returns:
            Latest transaction hash or None
        """
        try:
            variables = {
                "user": user_id,
                "first": 1,
                "skip": 0
            }
            
            client = self._create_client()
            result = client.execute(self.tx_query, variable_values=variables)
            transactions = result.get("userTransactions", [])
            
            if transactions:
                return transactions[0]["txHash"]
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get tx hash for user {user_id}: {e}")
            return None
    
    def fetch_positions(self, batch_size: int = 1000, max_positions: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch all Aave v3 positions with pagination.
        
        Args:
            batch_size: Number of positions to fetch per request
            max_positions: Maximum number of positions to fetch (None for all)
            
        Returns:
            DataFrame with position data
        """
        all_positions = []
        skip = 0
        
        logger.info("Starting to fetch Aave v3 positions...")
        
        while True:
            try:
                variables = {
                    "first": batch_size,
                    "skip": skip
                }
                
                logger.info(f"Fetching batch: skip={skip}, limit={batch_size}")
                
                client = self._create_client()
                result = client.execute(self.position_query, variable_values=variables)
                position_list = result.get("positions", [])
                
                if not position_list:
                    logger.info("No more positions to fetch")
                    break
                
                # Process the fetched positions
                processed_positions = self._process_position_data(position_list)
                all_positions.extend(processed_positions)
                
                logger.info(f"Fetched {len(position_list)} positions (total: {len(all_positions)})")
                
                # Check if we've reached the maximum
                if max_positions and len(all_positions) >= max_positions:
                    logger.info(f"Reached maximum positions limit: {max_positions}")
                    break
                
                # If we got fewer positions than requested, we've reached the end
                if len(position_list) < batch_size:
                    logger.info("Reached end of available positions")
                    break
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                skip += batch_size
                
            except Exception as e:
                logger.error(f"Error fetching batch at skip={skip}: {e}")
                # Continue with next batch
                skip += batch_size
                continue
        
        logger.info(f"Finished fetching positions. Total: {len(all_positions)}")
        
        # Create DataFrame with standardized columns
        df = pd.DataFrame(all_positions)
        
        # Ensure all required columns exist
        required_columns = [
            'id', 'user', 'market', 'reserveSymbol', 'assetId', 
            'hashOpened', 'hashClosed', 'balance', 'principal',
            'isCollateral', 'isIsolated', 'side', 'type', 'collateralBalanceETH'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Reorder columns to match expected format
        df = df[required_columns]
        
        # Print summary statistics
        logger.info(f"Summary:")
        logger.info(f"  Total positions: {len(df)}")
        logger.info(f"  Unique users: {df['user'].nunique()}")
        logger.info(f"  Unique assets: {df['reserveSymbol'].nunique()}")
        logger.info(f"  Total balance: {df['balance'].sum():.2f}")
        
        return df
    
    def save_positions(self, df: pd.DataFrame, filepath: Optional[str] = None) -> str:
        """
        Save positions DataFrame to CSV file.
        
        Args:
            df: DataFrame with position data
            filepath: Optional custom filepath
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aave_positions_{timestamp}.csv"
            filepath = os.path.join("data", filename)
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} positions to {filepath}")
        
        return filepath
    
    def _process_position_data(self, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw position data and convert to standardized format.
        
        Args:
            positions: List of raw position data from GraphQL
            
        Returns:
            List of processed position data
        """
        processed_positions = []
        
        for position in positions:
            try:
                # Extract basic position info
                position_id = position.get('id', '')
                account_id = position.get('account', {}).get('id', '')
                market_id = position.get('market', {}).get('id', '')
                asset_info = position.get('asset', {})
                asset_id = asset_info.get('id', '')
                asset_symbol = asset_info.get('symbol', '')
                
                # Get balance and principal
                balance = position.get('balance', '0')
                principal = position.get('principal', '0')
                
                # Convert string balances to float (they come as strings from GraphQL)
                try:
                    balance_float = float(balance) if balance else 0.0
                    principal_float = float(principal) if principal else 0.0
                except (ValueError, TypeError):
                    balance_float = 0.0
                    principal_float = 0.0
                
                # Skip positions with zero balance
                if balance_float <= 0:
                    continue
                
                # Create standardized position record
                processed_position = {
                    "id": position_id,
                    "user": account_id,
                    "market": market_id,
                    "reserveSymbol": asset_symbol,
                    "assetId": asset_id,
                    "hashOpened": position.get("hashOpened"),
                    "hashClosed": position.get("hashClosed"),
                    "balance": balance_float,
                    "principal": principal_float,
                    "isCollateral": position.get("isCollateral"),
                    "isIsolated": position.get("isIsolated"),
                    "side": position.get("side"),
                    "type": position.get("type"),
                    # Add placeholder for ETH equivalent (will be calculated later if needed)
                    "collateralBalanceETH": 0.0
                }
                
                processed_positions.append(processed_position)
                
            except Exception as e:
                logger.warning(f"Failed to process position {position.get('id', 'unknown')}: {e}")
                continue
        
        return processed_positions


def main():
    """Main function to fetch and save Aave positions."""
    try:
        # Initialize fetcher
        fetcher = AavePositionFetcher(rate_limit_delay=0.2, max_retries=3)
        
        # Fetch positions (limit to 10000 for testing, remove max_positions for all data)
        logger.info("Starting Aave position fetch...")
        df = fetcher.fetch_positions(batch_size=500, max_positions=10000)
        
        if df is None or df.empty:
            logger.warning("No positions found")
            return
        
        # Save positions to file
        filepath = fetcher.save_positions(df)
        
        # Display summary
        logger.info(f"Successfully saved positions to {filepath}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main() 