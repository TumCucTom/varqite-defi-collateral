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

```
> “Add to `varqite_partition.py`:  
> * Compute **edge-cut** and **modularity** before vs. after partitioning.  
> * Print percentage reduction.”  
```

```
> “Create `stress_local.py` that:  
> 1. Loads `labels.npy`.  
> 2. For each partition, runs a Monte-Carlo price-shock on the assets inside, recalculating health factors.  
> 3. Outputs a CSV `partition_risk.csv` with columns `clusterId, atRiskUSD`.  
> Keep the shocks simple (e.g., multivariate normal with 30 % st-dev and 0.8 intra-cluster correlation).”
```

```
> “Build a **Streamlit** app `app.py` that:  
> * Shows a Sankey or bipartite graph coloured by partitions.  
> * Plots `partition_risk.csv` as a bar chart.  
> * Has a sidebar slider for shock severity (re-runs `stress_local.py`).”
```

```
> “In `varqite_partition.py`, add a CLI flag `--ionq`.  
> If true:  
> 1. Import `IonQProvider` and route to `"ionq_aria"` backend.  
> 2. Set `shots=1000`, `optimization_level=3`.  
> 3. Print the returned job ID and polling status.  
> Else, default to Aer.”```

```
> “Create `benchmark.py` that:  
> * Runs partitioning twice (simulator vs. IonQ simulator).  
> * Logs wall-clock, circuit depth, # CNOTs, and partition quality metrics to a Markdown file `BENCH.md`.  
> Use Qiskit transpiler analysis passes.”  
```