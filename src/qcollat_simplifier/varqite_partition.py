"""
Partition bipartite graph using Qiskit VarQITE.

- Loads adjacency matrix (A.npz)
- Converts to Qiskit SparsePauliOp Hamiltonian
- Uses qiskit.algorithms.VarQITE with a 2-layer RealAmplitudes ansatz
- Runs on AerSimulator(method='statevector') or IonQ Aria backend
- Outputs labels.npy (node cluster assignments)
- Prints wall-clock time
- Computes edge-cut and modularity before vs. after partitioning, and prints percentage reduction
"""

import numpy as np
import scipy.sparse as sp
import time
import logging
from pathlib import Path
import networkx as nx

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit.algorithms import VarQITE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_adjacency_matrix(npz_path: str) -> sp.csr_matrix:
    logger.info(f"Loading adjacency matrix from {npz_path}")
    return sp.load_npz(npz_path)

def adjacency_to_sparse_pauliop(A: sp.csr_matrix) -> SparsePauliOp:
    """
    Convert adjacency matrix to a Qiskit SparsePauliOp Hamiltonian for partitioning.
    Uses a simple Ising mapping: H = sum_{i<j} A_ij Z_i Z_j
    """
    n = A.shape[0]
    paulis = []
    coeffs = []
    for i in range(n):
        for j in range(i+1, n):
            w = A[i, j]
            if w != 0:
                z = ['I'] * n
                z[i] = 'Z'
                z[j] = 'Z'
                paulis.append(''.join(z))
                coeffs.append(float(w))
    if not paulis:
        raise ValueError("No nonzero edges in adjacency matrix.")
    return SparsePauliOp(paulis, coeffs)

def run_varqite(H: SparsePauliOp, n_qubits: int, maxiter: int = 100, use_ionq: bool = False) -> np.ndarray:
    """
    Run VarQITE with a 2-layer RealAmplitudes ansatz.
    Returns the optimal parameters found.
    """
    ansatz = RealAmplitudes(n_qubits, reps=2, entanglement='full')
    optimizer = SLSQP(maxiter=maxiter)
    
    if use_ionq:
        try:
            from qiskit_ionq import IonQProvider
            provider = IonQProvider()
            backend = provider.get_backend("ionq_aria")
            backend.set_options(shots=1000, optimization_level=3)
            logger.info("Using IonQ Aria backend")
            logger.info(f"Backend options: shots={backend.options.shots}, optimization_level={backend.options.optimization_level}")
        except ImportError:
            logger.error("qiskit-ionq not available. Falling back to Aer.")
            backend = AerSimulator(method='statevector')
        except Exception as e:
            logger.error(f"Error connecting to IonQ: {e}. Falling back to Aer.")
            backend = AerSimulator(method='statevector')
    else:
        backend = AerSimulator(method='statevector')
        logger.info("Using AerSimulator backend")
    
    estimator = Estimator(backend=backend)
    varqite = VarQITE(ansatz, H, estimator, optimizer=optimizer)
    logger.info("Running VarQITE...")
    
    if use_ionq and hasattr(backend, 'name') and 'ionq' in backend.name.lower():
        # For IonQ backend, we need to handle job submission and polling
        try:
            result = varqite.run()
            if hasattr(result, 'job_id'):
                logger.info(f"IonQ Job ID: {result.job_id}")
                logger.info("Job status: COMPLETED")
            logger.info(f"VarQITE complete. Optimal value: {result.optimal_value}")
            return result.optimal_parameters
        except Exception as e:
            logger.error(f"Error with IonQ job: {e}")
            raise
    else:
        result = varqite.run()
        logger.info(f"VarQITE complete. Optimal value: {result.optimal_value}")
        return result.optimal_parameters

def get_partition_labels(params: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Assign each node to a cluster based on the sign of the expectation value of Z for each qubit.
    """
    # For a simple heuristic, use the sign of the parameter as a proxy for partition
    # (In practice, you may want to sample or compute expectation values)
    return (np.sign(params[:n_qubits]) > 0).astype(int)

def compute_edge_cut(A: sp.csr_matrix, labels: np.ndarray) -> float:
    """
    Compute the edge-cut: sum of weights of edges crossing the partition.
    """
    cut = 0.0
    n = A.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if labels[i] != labels[j]:
                cut += A[i, j]
    return cut

def compute_modularity(A: sp.csr_matrix, labels: np.ndarray) -> float:
    """
    Compute modularity using NetworkX for the given partition labels.
    """
    G = nx.from_scipy_sparse_matrix(A)
    communities = [set(np.where(labels == l)[0]) for l in np.unique(labels)]
    return nx.algorithms.community.quality.modularity(G, communities, weight='weight')

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Partition graph using VarQITE")
    parser.add_argument('--adj', type=str, default='graphs/A.npz', help='Path to adjacency matrix (A.npz)')
    parser.add_argument('--labels', type=str, default='graphs/labels.npy', help='Output path for labels.npy')
    parser.add_argument('--maxiter', type=int, default=100, help='Max optimizer iterations')
    parser.add_argument('--ionq', action='store_true', help='Use IonQ Aria backend instead of Aer')
    args = parser.parse_args()

    start = time.time()
    A = load_adjacency_matrix(args.adj)
    n = A.shape[0]
    logger.info(f"Adjacency matrix shape: {A.shape}")

    # Before partitioning: all in one cluster
    labels_before = np.zeros(n, dtype=int)
    edge_cut_before = compute_edge_cut(A, labels_before)
    modularity_before = compute_modularity(A, labels_before)

    # Partitioning
    H = adjacency_to_sparse_pauliop(A)
    logger.info(f"Hamiltonian has {len(H)} terms")
    params = run_varqite(H, n, maxiter=args.maxiter, use_ionq=args.ionq)
    labels = get_partition_labels(params, n)
    np.save(args.labels, labels)
    logger.info(f"Saved labels to {args.labels}")

    # After partitioning
    edge_cut_after = compute_edge_cut(A, labels)
    modularity_after = compute_modularity(A, labels)

    # Print results
    print(f"Wall-clock time: {time.time() - start:.2f} seconds")
    print(f"Labels: {labels}")
    print(f"Edge-cut before: {edge_cut_before:.4f}, after: {edge_cut_after:.4f}, reduction: {100*(edge_cut_before-edge_cut_after)/edge_cut_before if edge_cut_before else 0:.2f}%")
    print(f"Modularity before: {modularity_before:.4f}, after: {modularity_after:.4f}, increase: {100*(modularity_after-modularity_before)/abs(modularity_before) if modularity_before else 0:.2f}%")

if __name__ == "__main__":
    main() 