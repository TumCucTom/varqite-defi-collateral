# 1

```
from qiskit import Aer
backend = Aer.get_backend("aer_simulator_statevector")
```

to

```
from qiskit_ionq import IonQProvider          # new
provider = IonQProvider(token="YOUR_IONQ_KEY") # new
backend = provider.get_backend("ionq_simulator")  # or "ionq_aria"
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
> Else, default to Aer.” :contentReference[oaicite:2]{index=2} 
```

```
> “Create `benchmark.py` that:  
> * Runs partitioning twice (simulator vs. IonQ simulator).  
> * Logs wall-clock, circuit depth, # CNOTs, and partition quality metrics to a Markdown file `BENCH.md`.  
> Use Qiskit transpiler analysis passes.”
```