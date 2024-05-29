import math
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset
import json
import pyg_lib
from tqdm import tqdm


K = 5
DATASET_NAME = 'ogbn-arxiv'  # 'ogbn-products'  # 'ogbn-arxiv'
pt_file_path = '/Data/{DATASET_NAME}/all-roberta-large-v1/main/cached_embs/x_embs.pt'


def load_dataset(pt_file_path, set):
    embeddings = torch.load(pt_file_path)
    dataset = PygNodePropPredDataset(name=DATASET_NAME)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    set_idx = split_idx[set] # ['train'] ['valid'] ['test']
    return embeddings, set_idx, data


def compute_similarity(embeddings, node, neighbors):

    node_embedding = embeddings[node] 
    neighbor_embeddings = embeddings[neighbors]

    node_embedding = node_embedding / node_embedding.norm(dim=1, keepdim=True)
    neighbor_embeddings = neighbor_embeddings / neighbor_embeddings.norm(dim=1, keepdim=True)

    sim_scores = torch.mm(node_embedding, neighbor_embeddings.t()).squeeze(0)
    return sim_scores


def get_2hop_neighbors(data, set_idx):
    data.edge_index = to_undirected(data.edge_index)
    loader = NeighborLoader(data, input_nodes=set_idx, num_neighbors=[-1, -1], batch_size=1, shuffle=False)

    two_hop_neighbors = {}
    for batch in tqdm(loader, desc="Processing nodes"):
        if batch.edge_index.size(1) == 0:
            continue  

        node = batch.input_id
        neighbors = batch.n_id.tolist()

        if node in neighbors:
            neighbors.remove(node)  
        two_hop_neighbors[node] = neighbors

    return two_hop_neighbors


def find_top_k_neighbors(embeddings, two_hop_neighbors, k):
    top_k_neighbors = {}
    for node, neighbors in tqdm(two_hop_neighbors.items(), desc="Finding top-k neighbors"):
        if len(neighbors) >= k:
           
            neighbor_sims = compute_similarity(embeddings, node, neighbors)
            top_k = torch.topk(neighbor_sims, k=k, largest=True)
            top_k_neighbors[node] = [neighbors[i] for i in top_k.indices.tolist()]
        elif len(neighbors) > 0:
           
            neighbor_sims = compute_similarity(embeddings, node, neighbors)
            top_k = torch.topk(neighbor_sims, k=len(neighbors), largest=True)
            sorted_neighbors = [neighbors[i] for i in top_k.indices.tolist()]

            repeats_per_neighbor = math.ceil(k / len(sorted_neighbors))
            extended_neighbors = []
            for neighbor in sorted_neighbors:
                extended_neighbors.extend([neighbor] * repeats_per_neighbor)

            top_k_neighbors[node] = extended_neighbors[:k]
        else:
   
            top_k_neighbors[node] = [node] * k
            print(top_k_neighbors[node])

    return top_k_neighbors


def prepare_for_json(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if isinstance(key, torch.Tensor):
         
                new_key = key.item() if key.numel() == 1 else key.tolist()
            else:
                new_key = str(key)
            new_dict[new_key] = prepare_for_json(value)
        return new_dict
    elif isinstance(data, list):
        return [prepare_for_json(element) for element in data]
    elif isinstance(data, torch.Tensor):
        return data.tolist()
    else:
        return data


def save_to_json(data, file_name):
    data = prepare_for_json(data)
    print("save begin")
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)


def save_to_pt(data, file_name):
    torch.save(data, file_name)


def extract(set):
    embeddings, set_idx, data = load_dataset(pt_file_path, set)
    two_hop_neighbors = get_2hop_neighbors(data, set_idx)
    save_to_json(two_hop_neighbors , f'/Data/{DATASET_NAME}/all-roberta-large-v1/main/cached_embs/{set}_neighbors.json')
    save_to_pt(two_hop_neighbors, f'/Data/{DATASET_NAME}/all-roberta-large-v1/main/cached_embs/{set}_neighbors.pt')
    
    # two_hop_neighbors = torch.load(f'/Data/ogbn-products/all-roberta-large-v1/main/cached_embs/{set}_neighbors.pt')
    k = K  
    top_k_neighbors = find_top_k_neighbors(embeddings, two_hop_neighbors, k)
    save_to_json(top_k_neighbors, f'/Data/{DATASET_NAME}/all-roberta-large-v1/main/cached_embs/{set}_top_k_neighbors.json')
    save_to_pt(top_k_neighbors, f'/Data/{DATASET_NAME}/all-roberta-large-v1/main/cached_embs/{set}_top_k_neighbors.pt')


def main():
    extract('train')
    extract('valid')
    extract('test')


if __name__ == "__main__":
    main()
