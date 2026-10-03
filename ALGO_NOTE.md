# Algorithm Note: Vanilla Graph Convolutional Network

## 1. Problem

Given a graph G = (V, E), a node-feature matrix X, and labels for some nodes, learn node representations that use both feature information and graph structure.

## 2. Notation

- A: adjacency matrix, shape N x N
- I: identity matrix
- A_hat = A + I: adjacency with self-loops
- D_hat: degree matrix of A_hat
- X = H^(0): input node features
- H^(l): hidden representation at layer l
- W^(l): trainable weight matrix
- sigma: nonlinear activation

## 3. Vanilla GCN equation

H^(l+1) = sigma(D_hat^(-1/2) A_hat D_hat^(-1/2) H^(l) W^(l))

Define:

A_norm = D_hat^(-1/2) A_hat D_hat^(-1/2)

Then:

H^(l+1) = sigma(A_norm H^(l) W^(l))

## 4. Formula-by-formula derivation

### Step 1: Add self-loops

A_hat = A + I

Reason: a node must retain its own information while receiving neighbor information.

### Step 2: Compute degrees

D_hat(ii) = sum_j A_hat(ij)

### Step 3: Symmetric normalization

A_norm = D_hat^(-1/2) A_hat D_hat^(-1/2)

Reason: prevents high-degree nodes from dominating aggregation and produces a balanced propagation operator.

### Step 4: Linear feature transformation

Z^(l) = H^(l) W^(l)

### Step 5: Neighborhood aggregation

M^(l) = A_norm Z^(l)

Each node receives a weighted combination of transformed features from itself and its neighbors.

### Step 6: Activation

H^(l+1) = ReLU(M^(l))

For the final layer, logits can be passed directly to cross-entropy loss.

### Step 7: Output probabilities

P = softmax(H^(L))

### Step 8: Supervised loss

L = -sum_c y_c log(P_c)

For semi-supervised node classification, the loss is calculated only on labeled training nodes.

## 5. Two-layer Vanilla GCN

H^(1) = ReLU(A_norm X W^(0))

Z = A_norm H^(1) W^(1)

P = softmax(Z)

## 6. Algorithm

Input: adjacency A, features X, labels y for training nodes

1. Compute A_hat = A + I.
2. Compute D_hat from row sums of A_hat.
3. Compute A_norm = D_hat^(-1/2) A_hat D_hat^(-1/2).
4. Initialize W0 and W1.
5. Repeat for each training epoch:
   - H1 = ReLU(A_norm X W0)
   - Z = A_norm H1 W1
   - Compute cross-entropy on labeled training nodes.
   - Backpropagate gradients.
   - Update W0 and W1 using Adam.
6. Predict each node class using argmax(Z).

## 7. Tiny worked example

For a two-node connected graph:

A = [[0, 1],
     [1, 0]]

After self-loops:

A_hat = [[1, 1],
         [1, 1]]

Each degree is 2, so:

D_hat^(-1/2) = [[1/sqrt(2), 0],
                 [0, 1/sqrt(2)]]

Therefore:

A_norm = [[0.5, 0.5],
          [0.5, 0.5]]

Each node receives equal contribution from itself and its neighbor.

## 8. Complexity

Let N be the number of nodes, E the number of edges, F_in the input feature dimension, and F_out the output dimension.

A sparse GCN layer is approximately:

O(E * F_in + N * F_in * F_out)

For dense adjacency multiplication, the graph propagation term can become O(N^2 * F_in), which is why practical GCN systems use sparse representations for large graphs.

## 9. Strengths

- Simple and mathematically interpretable.
- Uses graph structure and features together.
- Parameter sharing makes it efficient compared with learning node-specific parameters.
- Strong baseline for node classification.

## 10. Limitations

- Repeated layers can cause over-smoothing.
- Vanilla GCN is transductive in its classic formulation.
- Large dense adjacency matrices are expensive.
- All neighbors are combined using graph-normalization weights rather than learned attention scores.

## 11. Implementation mapping

- A_hat = A + I -> `add_self_loops`
- D_hat -> `degree_matrix`
- D_hat^(-1/2) -> `inverse_sqrt_degree`
- A_norm -> `symmetric_normalize_adjacency`
- H W -> `linear_feature_transform`
- A_norm Z -> `neighborhood_aggregation`
- Complete layer -> `gcn_layer_formula`
- ReLU -> `relu`
- Softmax -> `softmax`
- Cross entropy -> `cross_entropy_loss`
