"""
Command-line interface for the Aave position fetcher.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .fetch_aave_positions import AavePositionFetcher
from .build_graph import AaveGraphBuilder
from .config import (
    DEFAULT_RATE_LIMIT_DELAY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_POSITIONS,
    CSV_OUTPUT_DIR,
    LOG_LEVEL,
    LOG_FORMAT
)

def setup_logging(level: str = LOG_LEVEL, format_str: str = LOG_FORMAT):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str
    )

def create_output_dir(output_dir: str):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch Aave v3 positions from Polygon subgraph and build graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    
    # Fetch command parser
    fetch_parser = subparsers.add_parser("fetch", help="Fetch Aave positions")
    
    fetch_parser.add_argument(
        "--max-positions",
        type=int,
        default=DEFAULT_MAX_POSITIONS,
        help=f"Maximum number of positions to fetch (default: {DEFAULT_MAX_POSITIONS})"
    )
    
    fetch_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of positions per batch (default: {DEFAULT_BATCH_SIZE})"
    )
    
    fetch_parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=DEFAULT_RATE_LIMIT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_RATE_LIMIT_DELAY})"
    )
    
    fetch_parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retries for failed requests (default: {DEFAULT_MAX_RETRIES})"
    )
    
    fetch_parser.add_argument(
        "--output-file",
        type=str,
        help="Custom output filename (default: raw_positions_YYYYMMDD.csv)"
    )
    
    fetch_parser.add_argument(
        "--output-dir",
        type=str,
        default=CSV_OUTPUT_DIR,
        help=f"Output directory (default: {CSV_OUTPUT_DIR})"
    )
    
    fetch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but don't save to file"
    )
    
    # Build graph command parser
    build_parser = subparsers.add_parser("build", help="Build graph from CSV data")
    
    build_parser.add_argument(
        "--input-file",
        type=str,
        help="Path to input CSV file (default: latest in data/)"
    )
    
    build_parser.add_argument(
        "--weight-type",
        choices=['collateral', 'debt', 'combined'],
        default='collateral',
        help="Edge weight type (default: collateral)"
    )
    
    build_parser.add_argument(
        "--normalization",
        choices=['minmax', 'zscore', 'log', 'none'],
        default='minmax',
        help="Weight normalization method (default: minmax)"
    )
    
    build_parser.add_argument(
        "--min-weight",
        type=float,
        default=0.001,
        help="Minimum weight to include in graph (default: 0.001)"
    )
    
    build_parser.add_argument(
        "--output-name",
        type=str,
        help="Base name for output files (default: A_YYYYMMDD_HHMMSS)"
    )
    
    # Test command parser
    test_parser = subparsers.add_parser("test", help="Test subgraph connection")
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=LOG_LEVEL,
        help=f"Logging level (default: {LOG_LEVEL})"
    )
    
    return parser.parse_args()

def handle_fetch(args):
    """Handle the fetch command."""
    logger = logging.getLogger(__name__)
    
    # Create output directory
    create_output_dir(args.output_dir)
    
    # Initialize fetcher
    logger.info("Initializing Aave position fetcher...")
    fetcher = AavePositionFetcher(
        rate_limit_delay=args.rate_limit_delay,
        max_retries=args.max_retries
    )
    
    # Fetch positions
    logger.info("Starting position fetch...")
    df = fetcher.fetch_positions(
        batch_size=args.batch_size,
        max_positions=args.max_positions
    )
    
    if df.empty:
        logger.warning("No positions found")
        return 0
    
    # Display summary
    logger.info(f"Fetched {len(df)} positions")
    
    if args.dry_run:
        logger.info("Dry run mode - not saving to file")
        return 0
    
    # Save to CSV
    if args.output_file:
        filepath = os.path.join(args.output_dir, args.output_file)
    else:
        filepath = None
        
    saved_filepath = fetcher.save_positions(df, filepath)
    logger.info(f"Successfully saved positions to {saved_filepath}")
    
    return 0

def handle_build(args):
    """Handle the build command."""
    logger = logging.getLogger(__name__)
    
    # Initialize builder
    builder = AaveGraphBuilder()
    
    # Load data
    df = builder.load_data(Path(args.input_file) if args.input_file else None)
    
    # Build graph
    builder.build_bipartite_graph(
        df, 
        weight_type=args.weight_type,
        normalization=args.normalization,
        min_weight=args.min_weight
    )
    
    # Create adjacency matrix
    adj_matrix = builder.create_adjacency_matrix()
    
    # Save graph
    builder.save_graph(adj_matrix, args.output_name)
    
    logger.info("Successfully built and saved graph")
    
    return 0

def handle_test():
    """Handle the test command."""
    logger = logging.getLogger(__name__)
    
    from .test_subgraph import test_subgraph_connection, test_schema
    
    logger.info("Testing subgraph connection...")
    if test_subgraph_connection() and test_schema():
        logger.info("✓ All tests passed!")
        return 0
    else:
        logger.error("✗ Tests failed!")
        return 1

def main():
    """Main CLI function."""
    args = parse_arguments()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == "fetch":
            return handle_fetch(args)
        elif args.command == "build":
            return handle_build(args)
        elif args.command == "test":
            return handle_test()
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 