# optspec-liquidity-routing-qubo

Quantum-inspired QUBO + Simulated Annealing solver for QCentroid use case **853** — Optimized Liquidity Routing.

Open-source stack: **pyqubo + dwave-neal** (no commercial SDK). Per-constraint Lagrange penalties (arXiv 2403.06699 technique) + warm-start from a greedy router.

## Emits the same 6 KPIs as the classical sibling for apples-to-apples benchmark

A/B sister: [`CrlsK/optspec-liquidity-routing-classical`](https://github.com/CrlsK/optspec-liquidity-routing-classical)
