import numpy as np

# Step 1: Define a simple knowledge graph
graph = {
    'A': {'D', 'C'},
    'B': ('A', 'D'),
    'C': {'A','E'},
    'D': {'B','F'},
    'E': {'C', 'G'},
    'F': {'D'},
    'G': {'E'}
}

# Step 2: Define synthetic embeddings
np.random.seed(42)
entity_embeddings = {e: np.random.rand(4) for e in graph}
print(f'entity_embeddings: {entity_embeddings}')
user_embeddings = {
    'User1': np.random.rand(4),
    'User2': np.random.rand(4)
}

# Step 3: Define relation weights (simplified as 1.0)
relation_weights = {(e, n): 1.0 for e in graph for n in graph[e]}
print(f'relation_weights :{relation_weights}')

# Step 4: Define interaction matrix Y
Y = [('User1', 'A'), ('User2', 'C')]

# Step 5: Helper functions
def get_receptive_field(graph, target_entity, H):
    M = {H: set([target_entity])}
    for h in range(H - 1, -1, -1):
        M[h] = set(M[h + 1])
        for e in M[h + 1]:
            M[h].update(graph.get(e, set()))
    return M

def aggregate(neighbor_embeddings, current_embedding):
    if len(neighbor_embeddings) == 0:
        return current_embedding
    return np.mean(neighbor_embeddings + [current_embedding], axis=0)

def kgcn_predict(user_embedding, item_embedding):
    return np.dot(user_embedding, item_embedding)

# Step 6: Training function
def train_kgcn(Y, graph, entity_embeddings, user_embeddings, relation_weights, H=2, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        for (u, v) in Y:
            M = get_receptive_field(graph, v, H)
            e_u = {0: {e: entity_embeddings[e] for e in M[0]}}
            for h in range(1, H + 1):
                e_u[h] = {}
                for e in M[h]:
                    neighbors = graph.get(e, set())
                    neighbor_embs = [e_u[h - 1].get(e_prime, entity_embeddings[e_prime]) for e_prime in neighbors]
                    weighted_neighbors = [relation_weights.get((e, e_prime), 1.0) * emb for emb, e_prime in zip(neighbor_embs, neighbors)]
                    e_u[h][e] = aggregate(weighted_neighbors, e_u[h - 1].get(e, entity_embeddings[e]))
            v_u = e_u[H][v]
            u_emb = user_embeddings[u]
            y_hat = kgcn_predict(u_emb, v_u)
            y_true = 1
            error = y_hat - y_true
            grad = error * v_u
            user_embeddings[u] -= learning_rate * grad
            entity_embeddings[v] -= learning_rate * error * u_emb
    return lambda u, v: kgcn_predict(user_embeddings[u], entity_embeddings[v])

# Step 7: Train and predict
predict_fn = train_kgcn(Y, graph, entity_embeddings, user_embeddings, relation_weights)

# Step 8: Show predictions
for u in user_embeddings:
    print(f"\nPredictions for {u}:")
    for v in graph:
        score = predict_fn(u, v)
        print(f"  Item {v}: Score {round(score, 2)}")