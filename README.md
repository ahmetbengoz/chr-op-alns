# chr-op-alns

Congestion-aware synchronization in multi-agent human–robot collaborative order picking systems using an event-based reservation simulation and Adaptive Large Neighborhood Search (ALNS).

## Overview

This repository contains the full computational pipeline used in the study. The model integrates:

- Event-based simulation with node and edge reservations (capacity = 1)
- Explicit picker–robot synchronization at aisle-end handover points
- Batch-level decision making (batching, handover assignment, routing)
- ALNS with destroy/repair operators, adaptive weights, and simulated annealing

The approach captures congestion and coordination effects that are typically ignored in simplified models.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt

### 2. Run the full pipeline
python code/main_pipeline.py

### 3. Output Files

Running the code generates all results automatically:

### 4.Data

data/dataset_instances.xlsx
data/results_table.xlsx
data/Table1_method_comparison.xlsx
data/Table2_fleet_size.xlsx

### 5.Figures

figures/figure2_convergence.png
Figure 2. ALNS convergence (averaged across the six hard instances)
figures/figure3_fleet_size.png
Figure 3. Effect of robot fleet size on makespan (mean ± 95% confidence interval across 90 instances, cap = 1)

### 6.Logs

logs/run_log.txt

### 7.Experimental Design

The instance set consists of 90 scenarios:
Aisles: {10, 20, 40}
Items: {200, 1000}
Density: {low, medium, high}
Seeds: 5 per configuration
ALNS is applied to a subset of 6 challenging instances:
40 aisles
medium/high density
1000 items

### 8.Key Parameters
The results in the manuscript are generated with:
AISLE_LEN = 20
CAP = 1
SERVICE = 4
BATCH_SIZE = 4
V_PICK = 1.0
V_ROB = 1.2

### 9.Methods Compared
Integrated (synchronization + congestion)
NoCongestion
NoSync
Sequential
Integrated-ALNS (on hard subset)
Performance metrics:
Makespan
Total robot waiting time

### 10.Reproducibility

All random seeds are fixed (MASTER_SEED = 123)
The full pipeline (instance generation → simulation → ALNS → tables → figures) is deterministic
All results can be reproduced by running a single script

### 11.Notes
The model is simulation-based; no time-expanded MIP is solved
ALNS is used as a refinement layer on top of the integrated solution
Results highlight strong non-linear effects in robot fleet sizing due to congestion and synchronization

### 12.Authors

Ahmet Bengöz
NATO Support and Procurement Agency (NSPA), Luxembourg
ORCID: 0000-0002-5772-4734
