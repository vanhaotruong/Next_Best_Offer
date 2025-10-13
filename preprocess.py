import pandas as pd
import numpy as np

def compute():
    link = pd.read_csv('./data/Amazon-KG-5core-Books.link', sep="\t")
    entity_tokens = link['entity_id:token'].unique()

    graph = pd.read_csv('./data/Amazon-KG-5core-Books.kg', sep="\t")
    head_tokens = graph['head_id:token'].unique()
    tail_tokens = graph['tail_id:token'].unique()

    all_entity_tokens = np.unique(np.concatenate([entity_tokens, head_tokens, tail_tokens]))
    entity2id = {entity: idx for idx, entity in enumerate(all_entity_tokens)}
    # id2entity = {idx: entity for entity, idx in entity2id.items()}

    item2entity = dict(zip(link['item_id:token'], link['entity_id:token']))
    item2entity_id = {item: entity2id[entity] for item, entity in item2entity.items()}

    graph['head_id'] = graph['head_id:token'].map(entity2id)
    graph['relation_id'] = graph['relation_id:token'].astype('category').cat.codes    
    graph['tail_id'] = graph['tail_id:token'].map(entity2id)

    graph = graph.dropna()
    graph = graph.astype({'head_id': 'long', 'relation_id': 'long', 'tail_id': 'long'})
    graph = graph[['head_id', 'relation_id', 'tail_id',
             'head_id:token', 'relation_id:token', 'tail_id:token']]

    graph.to_csv('./data/processed_graph.csv', index=False)
    np.save('./data/processed_graph.npy', graph.to_numpy())

    ########################################################
    interactions = pd.read_csv('./data/Amazon-KG-5core-Books.inter', sep= '\t')
    user_tokens = interactions['user_id:token'].unique()
    user2id = {user: idx for idx, user in enumerate(user_tokens)}
    # id2user = {idx: user for user, idx in user2id.items()}

    interactions['entity_id:token'] = interactions['item_id:token'].map(item2entity)
    interactions['user_id'] = interactions['user_id:token'].map(user2id)
    interactions['entity_id'] = interactions['item_id:token'].map(item2entity_id)

    interactions = interactions.dropna()
    interactions = interactions.astype({'user_id': 'long', 'entity_id': 'long'})
    interactions = interactions[['user_id', 'entity_id', 
                                 'entity_id:token', 'item_id:token', 'rating:float']]
    interactions.to_csv('./data/processed_interactions.csv', index= False)
    np.save('./data/processed_interactions.npy', interactions.to_numpy())
    

