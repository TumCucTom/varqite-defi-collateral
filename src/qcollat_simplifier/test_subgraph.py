"""
Test script to verify Aave v3 Polygon subgraph connectivity and schema.
"""

import logging
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from .config import AAVE_V3_POLYGON_SUBGRAPH, GRAPH_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_subgraph_connection():
    """Test connection to Aave v3 Polygon subgraph."""
    
    # Aave v3 Polygon subgraph endpoint
    subgraph_url = AAVE_V3_POLYGON_SUBGRAPH
    
    # Prepare headers
    headers = {
        'User-Agent': 'qcollat-simplifier/0.1.0'
    }
    
    # Add authorization header if API key is provided
    if GRAPH_API_KEY:
        headers['Authorization'] = f'Bearer {GRAPH_API_KEY}'
    
    try:
        # Create transport
        transport = RequestsHTTPTransport(
            url=subgraph_url,
            headers=headers
        )
        
        # Create client without schema fetching
        client = Client(transport=transport, fetch_schema_from_transport=False)
        
        logger.info("Successfully connected to Aave v3 Polygon subgraph")
        
        # Test a simple query
        test_query = gql("""
            query TestQuery {
                positions(first: 1) {
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
        
        result = client.execute(test_query)
        logger.info("Test query successful")
        logger.info(f"Sample data: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to connect to subgraph: {e}")
        return False

def test_schema():
    """Test and display schema information."""
    
    subgraph_url = AAVE_V3_POLYGON_SUBGRAPH
    
    # Prepare headers
    headers = {
        'User-Agent': 'qcollat-simplifier/0.1.0'
    }
    
    # Add authorization header if API key is provided
    if GRAPH_API_KEY:
        headers['Authorization'] = f'Bearer {GRAPH_API_KEY}'
    
    try:
        transport = RequestsHTTPTransport(url=subgraph_url, headers=headers)
        client = Client(transport=transport, fetch_schema_from_transport=False)
        
        # Try to get schema manually
        introspection_query = gql("""
            query IntrospectionQuery {
                __schema {
                    queryType {
                        name
                    }
                    types {
                        name
                        kind
                    }
                }
            }
        """)
        
        result = client.execute(introspection_query)
        
        logger.info("Schema query successful")
        logger.info(f"Query type: {result.get('__schema', {}).get('queryType', {}).get('name', 'Unknown')}")
        
        # Check for required types
        types = result.get('__schema', {}).get('types', [])
        type_names = [t.get('name') for t in types if t.get('name')]
        
        required_types = ['UserReserve', 'Reserve', 'User', 'Price']
        for type_name in required_types:
            if type_name in type_names:
                logger.info(f"✓ Found type: {type_name}")
            else:
                logger.warning(f"✗ Missing type: {type_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to fetch schema: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing Aave v3 Polygon subgraph connection...")
    
    # Test connection
    if test_subgraph_connection():
        logger.info("✓ Connection test passed")
    else:
        logger.error("✗ Connection test failed")
        exit(1)
    
    # Test schema
    if test_schema():
        logger.info("✓ Schema test passed")
    else:
        logger.error("✗ Schema test failed")
        exit(1)
    
    logger.info("All tests passed! Subgraph is ready for use.") 