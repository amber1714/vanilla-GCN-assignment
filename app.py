import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import load_karate_club
from formulas import (
    add_self_loops,
    degree_matrix,
    inverse_sqrt_degree,
    symmetric_normalize_adjacency,
)
from train import train_model


st.set_page_config(page_title="Vanilla GCN Assignment", layout="wide")
st.title("Vanilla Graph Convolutional Network")
st.caption("Formula-by-formula implementation of the Vanilla GCN")

phase = st.sidebar.radio(
    "Choose phase",
    ["Phase 1: Understanding", "Phase 2: Mathematics", "Phase 3: Implementation"],
)

graph, adjacency, features, labels, train_idx, val_idx, test_idx = load_karate_club()

if phase == "Phase 1: Understanding":
    st.header("Basic understanding")
    st.write(
        "A Vanilla GCN learns node representations by combining each node's own features "
        "with information from neighboring nodes using a normalized adjacency matrix."
    )
    st.latex(r"H^{(l+1)} = \sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)})")
    st.write("Dataset: Zachary's Karate Club. Task: classify each member into one of two communities.")

    pos = nx.spring_layout(graph, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx(
        graph,
        pos=pos,
        node_color=labels.numpy(),
        cmap=plt.cm.Set2,
        with_labels=True,
        ax=ax,
    )
    ax.set_axis_off()
    st.pyplot(fig)

elif phase == "Phase 2: Mathematics":
    st.header("Mathematical pipeline")
    st.latex(r"\hat{A} = A + I")
    st.latex(r"\hat{D}_{ii} = \sum_j \hat{A}_{ij}")
    st.latex(r"A_{norm} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}")
    st.latex(r"Z^{(l)} = H^{(l)}W^{(l)}")
    st.latex(r"H^{(l+1)} = \sigma(A_{norm} Z^{(l)})")
    st.latex(r"\mathcal{L} = -\sum_{c=1}^{C} y_c \log \hat{y}_c")

    a_hat = add_self_loops(adjacency)
    d_hat = degree_matrix(a_hat)
    d_inv_sqrt = inverse_sqrt_degree(d_hat)
    a_norm = symmetric_normalize_adjacency(adjacency)

    choice = st.selectbox("Inspect a matrix", ["A", "A_hat", "D_hat", "D_hat^-1/2", "A_norm"])
    selected = {
        "A": adjacency,
        "A_hat": a_hat,
        "D_hat": d_hat,
        "D_hat^-1/2": d_inv_sqrt,
        "A_norm": a_norm,
    }[choice]
    st.dataframe(pd.DataFrame(selected.numpy()).round(4), use_container_width=True)

else:
    st.header("Implementation and training")
    epochs = st.slider("Epochs", 50, 500, 300, 50)
    lr = st.select_slider("Learning rate", options=[0.001, 0.005, 0.01, 0.02, 0.05], value=0.01)

    if st.button("Train Vanilla GCN"):
        with st.spinner("Training..."):
            model, predictions, test_acc = train_model(epochs=epochs, lr=lr)

        st.metric("Test accuracy", f"{test_acc * 100:.2f}%")
        results = pd.DataFrame(
            {
                "Node": list(range(len(labels))),
                "True label": labels.numpy(),
                "Predicted label": predictions.numpy(),
            }
        )
        st.dataframe(results, use_container_width=True)

        pos = nx.spring_layout(graph, seed=42)
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw_networkx(
            graph,
            pos=pos,
            node_color=predictions.numpy(),
            cmap=plt.cm.Set2,
            with_labels=True,
            ax=ax,
        )
        ax.set_title("Predicted Communities")
        ax.set_axis_off()
        st.pyplot(fig)
