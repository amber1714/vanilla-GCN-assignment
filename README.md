# Vanilla GCN Assignment

A complete three-phase implementation of a Vanilla Graph Convolutional Network (GCN), designed for an algorithm assignment and GitHub portfolio.

## Assignment phases

### Phase 1 — Basic understanding
A Graph Convolutional Network learns node representations by aggregating information from neighboring nodes. Vanilla GCN uses a normalized adjacency matrix so that information propagation remains numerically stable and each node also receives its own features through a self-loop.

Core update:

**H^(l+1) = sigma(D_hat^(-1/2) A_hat D_hat^(-1/2) H^l W^l)**

### Phase 2 — Mathematical algorithm note
See `ALGO_NOTE.md` for notation, derivation, algorithm steps, complexity, and a worked mini-example.

### Phase 3 — Formula-by-formula implementation
Each mathematical step is implemented separately in `src/formulas.py`.

## Project structure

- `ALGO_NOTE.md` — mathematical notes and algorithm
- `src/formulas.py` — each GCN formula as a Python function
- `src/model.py` — Vanilla GCN layer and 2-layer model
- `src/data.py` — Karate Club graph loader
- `src/train.py` — training and evaluation
- `app.py` — Streamlit interactive demo
- `tests/test_formulas.py` — correctness tests for core formulas
- `requirements.txt` — dependencies

## Dataset

The demo uses Zachary's Karate Club graph. Each node represents a club member and each edge represents a social relationship. The target is the known two-community split.

## Run

Create a Python environment, install `requirements.txt`, then run `src/train.py` for terminal training or `app.py` with Streamlit for the interactive interface.

## Deployment

The repository is ready for Streamlit Community Cloud, Railway, or another Python hosting platform. The app entrypoint is `app.py`.
