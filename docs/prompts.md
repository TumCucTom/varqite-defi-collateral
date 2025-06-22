```
pip project called qcollat-simplifier.
Add these packages:
qiskit, qiskit-algorithms, qiskit-aer, qiskit-ionq, networkx, numpy, scipy, pandas, gql, graphql-core, matplotlib, plotly, streamlit.
Include a requirements.txt or pyproject.toml with exact versions.”

Do not write any python scripts yet
```

```
Write fetch_aave_positions.py that:

Queries the Aave v3 Polygon subgraph (use GraphQL endpoint from The Graph).

Returns a Pandas DataFrame with columns
txHash, user, reserveSymbol, collateralBalanceETH, debtBalanceETH.

Save as raw_positions_YYYYMMDD.csv.
Include robust pagination and rate-limit handling.”
```

```
Create build_graph.py that:

Loads the latest CSV.

Uses NetworkX to build a bipartite graph:

asset nodes (ETH, WBTC, MATIC, …)

position nodes (one per txHash or user-reserve pair).

Edge weight = collateralBalanceETH (or debt) normalised.

Export adjacency matrix as a SciPy CSR sparse matrix A.npz.”
```

```
Write varqite_partition.py that:

Loads A.npz, converts to Qiskit SparsePauliOp Hamiltonian.

Uses qiskit.algorithms.VarQITE with a simple hardware-efficient ansatz (2-layer HEA, RealAmplitudes).

Runs on AerSimulator(method='statevector').

Outputs a NumPy array labels.npy assigning each node to a partition cluster.
Show wall-clock time.” 
```