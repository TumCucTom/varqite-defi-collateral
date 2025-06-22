"""
Configuration settings for the Aave position fetcher.
"""

# Subgraph endpoints
AAVE_V3_POLYGON_SUBGRAPH = "https://gateway.thegraph.com/api/subgraphs/id/6yuf1C49aWEscgk5n9D1DekeG1BCk5Z9imJYJT3sVmAT"

# Graph API key (required for gateway.thegraph.com endpoints)
GRAPH_API_KEY = "17ed8912f199196cd1085a764816a53c"

def set_api_key(api_key: str):
    """Set the Graph API key for subgraph access."""
    global GRAPH_API_KEY
    GRAPH_API_KEY = api_key

# Instructions for getting an API key:
# 1. Go to https://thegraph.com/studio/
# 2. Sign up or log in
# 3. Create a new API key
# 4. Use the key with: from qcollat_simplifier.config import set_api_key; set_api_key("your_api_key_here")

# Rate limiting and retry settings
DEFAULT_RATE_LIMIT_DELAY = 0.2  # seconds between requests
DEFAULT_MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_POSITIONS = 10000  # Set to None for all positions

# CSV output settings
CSV_OUTPUT_DIR = "data"
CSV_FILENAME_PREFIX = "raw_positions"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# GraphQL query settings
QUERY_TIMEOUT = 30  # seconds
MAX_CONCURRENT_REQUESTS = 1  # Set to 1 for sequential requests

# Data processing settings
MIN_COLLATERAL_ETH = 0.0  # Minimum collateral balance to include
MIN_DEBT_ETH = 0.0        # Minimum debt balance to include 