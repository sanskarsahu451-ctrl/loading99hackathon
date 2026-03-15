import numpy as np
import pyzx as zx

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

# DNA Sequence

dna = "ATGCGTACGTTAGCGTACGATCGTAGCTAGCTTGACGATCGTACGTTAGC"

# DNA Encoding

mapping = {'A':0,'T':1,'G':2,'C':3}
values = [mapping[b] for b in dna]

angles = [(v/3)*2*np.pi for v in values]

# Build Circuit

n_qubits = 5
qc = QuantumCircuit(n_qubits, n_qubits)

for i,angle in enumerate(angles):
    qc.ry(angle, i % n_qubits)

for i in range(n_qubits-1):
    qc.cx(i, i+1)

# ZX-CALCULUS OPTIMIZATION


from qiskit.qasm2 import dumps

qasm_str = dumps(qc)

zx_circ = zx.Circuit.from_qasm(qasm_str)

graph = zx_circ.to_graph()

zx.simplify.full_reduce(graph)

optimized = zx.extract_circuit(graph)

opt_qasm = optimized.to_qasm()

qc_opt = QuantumCircuit.from_qasm_str(opt_qasm)

qc_opt.measure_all()


# IDEAL SIMULATION (Aer)


ideal_backend = AerSimulator()

ideal_transpiled = transpile(qc_opt, ideal_backend)

ideal_job = ideal_backend.run(ideal_transpiled, shots=4096)

ideal_counts = ideal_job.result().get_counts()

# NOISY SIMULATION (FakeSherbrooke)


fake_backend = FakeSherbrooke()

noisy_backend = AerSimulator.from_backend(fake_backend)

noisy_transpiled = transpile(qc_opt, fake_backend, optimization_level=3)

noisy_job = noisy_backend.run(noisy_transpiled, shots=4096)

noisy_counts = noisy_job.result().get_counts()

# Convert Counts → Probabilities


def counts_to_prob(counts, shots):
    return {k:v/shots for k,v in counts.items()}

ideal_prob = counts_to_prob(ideal_counts,4096)
noisy_prob = counts_to_prob(noisy_counts,4096)

# Bhattacharyya Fidelity


keys = set(ideal_prob) | set(noisy_prob)

fid = sum(
    np.sqrt(ideal_prob.get(k,0)*noisy_prob.get(k,0))
    for k in keys
)

print("\nDistribution Fidelity:", fid)


# Circuit Metrics


depth = noisy_transpiled.depth()
ops = noisy_transpiled.count_ops()

swap_count = ops.get('swap',0)

two_qubit = 0
for inst,qargs,cargs in noisy_transpiled.data:
    if len(qargs)==2:
        two_qubit +=1

print("\nCircuit Metrics")
print("Depth:",depth)
print("Swap Gates:",swap_count)
print("Two-Qubit Gates:",two_qubit)

print("\nQuantum Circuit:")
print(qc_opt.draw())
