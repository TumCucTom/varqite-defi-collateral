"""
Benchmark script for quantum partitioning performance comparison.

Runs partitioning twice (simulator vs. IonQ simulator) and logs:
- Wall-clock time
- Circuit depth
- Number of CNOTs
- Partition quality metrics
"""

import numpy as np
import scipy.sparse as sp
import time
import logging
from pathlib import Path
import networkx as nx
from datetime import datetime
from typing import Dict, Any, Tuple

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit.algorithms import VarQITE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Depth, CountOps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Benchmark runner for quantum partitioning."""
    
    def __init__(self, output_file: str = "BENCH.md"):
        self.output_file = output_file
        self.results = {}
        
    def load_data(self, adj_path: str = "graphs/A.npz") -> Tuple[sp.csr_matrix, int]:
        """Load adjacency matrix data."""
        logger.info(f"Loading adjacency matrix from {adj_path}")
        A = sp.load_npz(adj_path)
        n = A.shape[0]
        logger.info(f"Matrix shape: {A.shape}, nodes: {n}")
        return A, n
    
    def adjacency_to_sparse_pauliop(self, A: sp.csr_matrix) -> SparsePauliOp:
        """Convert adjacency matrix to SparsePauliOp Hamiltonian."""
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
    
    def analyze_circuit(self, circuit, backend_name: str) -> Dict[str, Any]:
        """Analyze circuit using Qiskit transpiler passes."""
        logger.info(f"Analyzing circuit for {backend_name}")
        
        # Transpile circuit for the backend
        if 'ionq' in backend_name.lower():
            # Use IonQ backend for transpilation
            try:
                from qiskit_ionq import IonQProvider
                provider = IonQProvider()
                backend = provider.get_backend("ionq_aria")
            except:
                # Fallback to generic backend
                from qiskit.providers.fake_provider import FakeIonQ
                backend = FakeIonQ()
        else:
            backend = AerSimulator(method='statevector')
        
        # Transpile
        transpiled_circuit = transpile(circuit, backend, optimization_level=3)
        
        # Analyze with passes
        pm = PassManager([
            Depth(),
            CountOps()
        ])
        analysis_result = pm.run(transpiled_circuit)
        
        # Extract metrics
        depth = analysis_result.depth
        ops = analysis_result.count_ops()
        cnot_count = ops.get('cx', 0) + ops.get('cz', 0)  # Count both CX and CZ gates
        
        return {
            'depth': depth,
            'cnot_count': cnot_count,
            'total_ops': sum(ops.values()),
            'ops_breakdown': ops
        }
    
    def compute_partition_metrics(self, A: sp.csr_matrix, labels: np.ndarray) -> Dict[str, float]:
        """Compute partition quality metrics."""
        # Edge cut
        edge_cut = 0.0
        n = A.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if labels[i] != labels[j]:
                    edge_cut += A[i, j]
        
        # Modularity
        G = nx.from_scipy_sparse_matrix(A)
        communities = [set(np.where(labels == l)[0]) for l in np.unique(labels)]
        modularity = nx.algorithms.community.quality.modularity(G, communities, weight='weight')
        
        # Number of clusters
        n_clusters = len(np.unique(labels))
        
        # Cluster balance (standard deviation of cluster sizes)
        cluster_sizes = [np.sum(labels == cluster) for cluster in np.unique(labels)]
        cluster_balance = np.std(cluster_sizes) if len(cluster_sizes) > 1 else 0
        
        return {
            'edge_cut': edge_cut,
            'modularity': modularity,
            'n_clusters': n_clusters,
            'cluster_balance': cluster_balance,
            'avg_cluster_size': np.mean(cluster_sizes)
        }
    
    def run_partitioning(self, H: SparsePauliOp, n_qubits: int, backend_name: str, 
                        use_ionq: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run partitioning with specified backend and collect metrics."""
        logger.info(f"Running partitioning with {backend_name}")
        
        start_time = time.time()
        
        # Create ansatz
        ansatz = RealAmplitudes(n_qubits, reps=2, entanglement='full')
        
        # Analyze circuit before optimization
        circuit_metrics = self.analyze_circuit(ansatz, backend_name)
        
        # Setup backend
        if use_ionq:
            try:
                from qiskit_ionq import IonQProvider
                provider = IonQProvider()
                backend = provider.get_backend("ionq_aria")
                backend.set_options(shots=1000, optimization_level=3)
            except Exception as e:
                logger.warning(f"Could not connect to IonQ: {e}. Using Aer.")
                backend = AerSimulator(method='statevector')
        else:
            backend = AerSimulator(method='statevector')
        
        # Run VarQITE
        optimizer = SLSQP(maxiter=100)
        estimator = Estimator(backend=backend)
        varqite = VarQITE(ansatz, H, estimator, optimizer=optimizer)
        
        result = varqite.run()
        end_time = time.time()
        
        # Get partition labels
        params = result.optimal_parameters
        labels = (np.sign(params[:n_qubits]) > 0).astype(int)
        
        # Collect timing
        wall_clock_time = end_time - start_time
        
        return labels, {
            'wall_clock_time': wall_clock_time,
            'optimal_value': result.optimal_value,
            'circuit_metrics': circuit_metrics
        }
    
    def run_benchmark(self, adj_path: str = "graphs/A.npz"):
        """Run complete benchmark comparing simulator vs IonQ."""
        logger.info("Starting benchmark...")
        
        # Load data
        A, n = self.load_data(adj_path)
        H = self.adjacency_to_sparse_pauliop(A)
        
        # Run with Aer simulator
        logger.info("=== Running with Aer Simulator ===")
        labels_aer, metrics_aer = self.run_partitioning(H, n, "AerSimulator", use_ionq=False)
        partition_metrics_aer = self.compute_partition_metrics(A, labels_aer)
        
        # Run with IonQ simulator
        logger.info("=== Running with IonQ Simulator ===")
        labels_ionq, metrics_ionq = self.run_partitioning(H, n, "IonQ", use_ionq=True)
        partition_metrics_ionq = self.compute_partition_metrics(A, labels_ionq)
        
        # Store results
        self.results = {
            'aer': {
                'labels': labels_aer,
                'metrics': metrics_aer,
                'partition_metrics': partition_metrics_aer
            },
            'ionq': {
                'labels': labels_ionq,
                'metrics': metrics_ionq,
                'partition_metrics': partition_metrics_ionq
            },
            'timestamp': datetime.now().isoformat(),
            'problem_size': n
        }
        
        logger.info("Benchmark completed!")
    
    def save_results(self):
        """Save benchmark results to Markdown file."""
        logger.info(f"Saving results to {self.output_file}")
        
        with open(self.output_file, 'w') as f:
            f.write("# Quantum Partitioning Benchmark Results\n\n")
            f.write(f"**Timestamp:** {self.results['timestamp']}\n")
            f.write(f"**Problem Size:** {self.results['problem_size']} nodes\n\n")
            
            # Performance Comparison
            f.write("## Performance Comparison\n\n")
            f.write("| Metric | Aer Simulator | IonQ Simulator | Difference |\n")
            f.write("|--------|---------------|----------------|------------|\n")
            
            aer_time = self.results['aer']['metrics']['wall_clock_time']
            ionq_time = self.results['ionq']['metrics']['wall_clock_time']
            time_diff = ((ionq_time - aer_time) / aer_time) * 100
            
            f.write(f"| Wall-clock Time (s) | {aer_time:.2f} | {ionq_time:.2f} | {time_diff:+.1f}% |\n")
            
            aer_depth = self.results['aer']['metrics']['circuit_metrics']['depth']
            ionq_depth = self.results['ionq']['metrics']['circuit_metrics']['depth']
            depth_diff = ((ionq_depth - aer_depth) / aer_depth) * 100 if aer_depth > 0 else 0
            
            f.write(f"| Circuit Depth | {aer_depth} | {ionq_depth} | {depth_diff:+.1f}% |\n")
            
            aer_cnot = self.results['aer']['metrics']['circuit_metrics']['cnot_count']
            ionq_cnot = self.results['ionq']['metrics']['circuit_metrics']['cnot_count']
            cnot_diff = ((ionq_cnot - aer_cnot) / aer_cnot) * 100 if aer_cnot > 0 else 0
            
            f.write(f"| CNOT Count | {aer_cnot} | {ionq_cnot} | {cnot_diff:+.1f}% |\n")
            
            # Partition Quality Comparison
            f.write("\n## Partition Quality Comparison\n\n")
            f.write("| Metric | Aer Simulator | IonQ Simulator | Difference |\n")
            f.write("|--------|---------------|----------------|------------|\n")
            
            aer_edge_cut = self.results['aer']['partition_metrics']['edge_cut']
            ionq_edge_cut = self.results['ionq']['partition_metrics']['edge_cut']
            edge_cut_improvement = ((aer_edge_cut - ionq_edge_cut) / aer_edge_cut) * 100 if aer_edge_cut > 0 else 0
            
            f.write(f"| Edge Cut | {aer_edge_cut:.4f} | {ionq_edge_cut:.4f} | {edge_cut_improvement:+.1f}% |\n")
            
            aer_modularity = self.results['aer']['partition_metrics']['modularity']
            ionq_modularity = self.results['ionq']['partition_metrics']['modularity']
            modularity_improvement = ((ionq_modularity - aer_modularity) / abs(aer_modularity)) * 100 if aer_modularity != 0 else 0
            
            f.write(f"| Modularity | {aer_modularity:.4f} | {ionq_modularity:.4f} | {modularity_improvement:+.1f}% |\n")
            
            aer_clusters = self.results['aer']['partition_metrics']['n_clusters']
            ionq_clusters = self.results['ionq']['partition_metrics']['n_clusters']
            
            f.write(f"| Number of Clusters | {aer_clusters} | {ionq_clusters} | {ionq_clusters - aer_clusters:+d} |\n")
            
            # Detailed Circuit Analysis
            f.write("\n## Detailed Circuit Analysis\n\n")
            
            for backend_name, data in [('Aer Simulator', self.results['aer']), ('IonQ Simulator', self.results['ionq'])]:
                f.write(f"### {backend_name}\n\n")
                circuit_metrics = data['metrics']['circuit_metrics']
                
                f.write(f"- **Circuit Depth:** {circuit_metrics['depth']}\n")
                f.write(f"- **CNOT Count:** {circuit_metrics['cnot_count']}\n")
                f.write(f"- **Total Operations:** {circuit_metrics['total_ops']}\n")
                f.write(f"- **Operations Breakdown:** {circuit_metrics['ops_breakdown']}\n")
                f.write(f"- **Wall-clock Time:** {data['metrics']['wall_clock_time']:.2f}s\n")
                f.write(f"- **Optimal Value:** {data['metrics']['optimal_value']:.4f}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write("This benchmark compares the performance of quantum partitioning using ")
            f.write("Aer simulator vs IonQ simulator. Key findings:\n\n")
            
            if time_diff > 0:
                f.write(f"- IonQ simulator is {time_diff:.1f}% slower than Aer simulator\n")
            else:
                f.write(f"- IonQ simulator is {abs(time_diff):.1f}% faster than Aer simulator\n")
            
            if edge_cut_improvement > 0:
                f.write(f"- IonQ achieves {edge_cut_improvement:.1f}% better edge cut\n")
            else:
                f.write(f"- Aer achieves {abs(edge_cut_improvement):.1f}% better edge cut\n")
            
            if modularity_improvement > 0:
                f.write(f"- IonQ achieves {modularity_improvement:.1f}% better modularity\n")
            else:
                f.write(f"- Aer achieves {abs(modularity_improvement):.1f}% better modularity\n")
        
        logger.info(f"Results saved to {self.output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark quantum partitioning performance")
    parser.add_argument('--adj', type=str, default='graphs/A.npz', 
                       help='Path to adjacency matrix (A.npz)')
    parser.add_argument('--output', type=str, default='BENCH.md',
                       help='Output Markdown file (default: BENCH.md)')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = BenchmarkRunner(output_file=args.output)
    benchmark.run_benchmark(adj_path=args.adj)
    benchmark.save_results()
    
    print(f"\nBenchmark completed! Results saved to {args.output}")
    print("Check the Markdown file for detailed performance comparison.")


if __name__ == "__main__":
    main() 