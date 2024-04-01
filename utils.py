import numpy as np
import os
import networkx as nx
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from ogb.nodeproppred import Evaluator
import random
from ogb.nodeproppred import NodePropPredDataset


def getFilesInDirectory(directory):
    fileList = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            fileList.append(file)
    return fileList

def saveLog(log, logFile=None):
    saveFolder = os.getenv('SAVE_FOLDER', '.')
    logFile = logFile or f'{saveFolder}/log.txt'
    with open(logFile, 'a') as f:
        f.write(f"{log}\n")
    print(log)
    

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        saveLog(f"Created folder: {path}")

def rotate_matrix(matrix, rotation):
    return np.dot(matrix, rotation)

def project_onto_column_space(matrix_a, vec):
    return matrix_a @ np.linalg.lstsq(matrix_a, vec, rcond=None)[0]

def DVCS(D):
    M, N = D.shape
    assert N % 2 == 0, "N should be even"
    Dout = np.zeros((M, N // 2))
    for i in range(M):
        for j in range(N // 2):
            Dout[i,j] = 0.5 * D[2*j,2*j+1] * (D[i,2*j] - D[i,2*j+1]) * (D[i,2*j] + D[i,2*j+1])
    return Dout

def load_datasets(dataset_name, root_dir='dataset/'):
    data, label = {}, {}
    dataset = NodePropPredDataset(name=dataset_name, root=root_dir)
    data[dataset_name] = dataset[0][0]
    label[dataset_name] = dataset[0][1]
    label[dataset_name] = torch.from_numpy(label[dataset_name]).to(torch.float64)
    return data, label, dataset

def log_dataset_info(data, label, name):
    saveLog(name)
    evaluator = Evaluator(name=name)
    #log_evaluator_info(evaluator)
    #log_data_info(data[name], label[name])

def log_evaluator_info(evaluator):
    saveLog(evaluator.expected_input_format) 
    saveLog(evaluator.expected_output_format) 

def log_data_info(data, label):
    saveLog(data.keys())
    saveLog(data['num_nodes'])
    saveLog(data['edge_index'].shape)
    saveLog(label.shape)
    saveLog(label[:5,:])
    log_feature_info(data, 'edge_feat')
    log_feature_info(data, 'node_feat', alt_key='node_species')

def log_feature_info(data, key, alt_key=None):
    if data[key] is not None:
        saveLog(data[key].shape)
    elif alt_key and data[alt_key] is not None:
        saveLog(data[alt_key].shape)
    else:
        saveLog("unweighted")


def initialize_graph(data, name):
    """Initialize a NetworkX graph with nodes and edges."""
    e1 = data[name]['edge_index'][0, :]
    e2 = data[name]['edge_index'][1, :]
    n = data[name]['num_nodes']
    nodes = list(range(n))
    edges = list(zip(e1, e2))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def analyze_components(G, share_data_folder):
    """Analyze and save the largest component of the graph."""
    if os.path.exists(f"{share_data_folder}/maxLenComponentIndex.pt"):
        saveLog("Largest component already exists, skipping...")
        return torch.load(f"{share_data_folder}/maxLenComponentIndex.pt")
    connected_components = list(nx.connected_components(G))
    
    # Find the largest component
    largest_component = max(connected_components, key=len)
    largest_component = np.array(list(largest_component))
    save_component_data(largest_component, share_data_folder)

    return largest_component

def save_component_data(largest_component, share_data_folder):
    """Save the largest component and related data."""
    filename = f"{share_data_folder}/maxLenComponentIndex.pt"
    if not os.path.exists(filename):
        torch.save(torch.from_numpy(np.array(list(largest_component))), filename)
        
def handle_indices(dataset, share_data_folder, component):
    """Handle train, test, and validation indices based on the component."""
    if os.path.exists(f"{share_data_folder}/train_idx.pt"):
        saveLog("Indices already exist, skipping...")
        return
    split_idx = dataset.get_idx_split()
    save_indices(split_idx, share_data_folder, suffix="original")
    
    if len(component) > 0:
        
        # Filter indices to only include those in the largest component
        filtered_indices = {split: [idx for idx in split_idx[split] if idx in component] for split in ["train", "valid", "test"]}
        save_filtered_indices(filtered_indices, share_data_folder)

def save_indices(indices, share_data_folder, suffix=""):
    """Save the train, valid, and test indices."""
    for key in indices:
        filename = f"{share_data_folder}/{key}_idx_{suffix}.pt"
        torch.save(indices[key], filename)

def save_filtered_indices(filtered_indices, share_data_folder):
    """Save filtered train, valid, and test indices after component analysis."""
    for split, idx in filtered_indices.items():
        filename = f"{share_data_folder}/{split}_idx.pt"
        torch.save(torch.from_numpy(np.array(idx)), filename)

def load_and_save_split_indices(dataset, shareDataFolder):
    """Load split indices from the dataset and save them."""
    split_idx = dataset.get_idx_split()
    # Assumes `dataset` is predefined or passed as a parameter correctly
    del dataset  # Ensure dataset is no longer needed before deleting
    
    # Save original indices
    torch.save(split_idx["test"], f"{shareDataFolder}/test_idx_original.pt")
    torch.save(split_idx["valid"], f"{shareDataFolder}/valid_idx_original.pt")
    torch.save(torch.from_numpy(np.array(split_idx["train"])), f"{shareDataFolder}/train_idx.pt")
    torch.save(torch.from_numpy(np.array(split_idx["test"])), f"{shareDataFolder}/test_idx.pt")
    torch.save(torch.from_numpy(np.array(split_idx["valid"])), f"{shareDataFolder}/valid_idx.pt")
    
    log_index_shapes(split_idx["train"], split_idx["test"], split_idx["valid"])

def log_index_shapes(train_idx, test_idx, valid_idx):
    """Log the shapes of train, test, and valid indices."""
    saveLog(f"shape of train_idx: {len(train_idx)}")
    saveLog(f"shape of test_idx: {len(test_idx)}")
    saveLog(f"shape of valid_idx: {len(valid_idx)}")

def load_or_generate_anchor_indices(share_data_folder, component, G, num_anchors, name, data):
    file_path = os.path.join(share_data_folder, 'anchorIndicesList.npy')
    if os.path.exists(file_path):
        anchor_indices_list = np.load(file_path, allow_pickle=True)
    else:
        anchor_indices_list = generate_anchor_indices(component, G, num_anchors, name, data)
    np.save(file_path, anchor_indices_list)
    return anchor_indices_list

def generate_anchor_indices(component, G, num_anchors, name, data):
    if name == "ogbn-proteins":
        anchor_indices_list = np.random.choice(data[name]['num_nodes'], num_anchors, replace=False)
    elif name == "ogbn-products":
        #anchor_indices_list = [np.random.choice(component, num_anchors).reshape(-1)]
        anchor_indices_list = np.random.choice(component, num_anchors)
        #anchor_indices_list = np.array([anchor_indices_list])
    return np.array(anchor_indices_list)

def log_num_anchors(anchor_indices_list):
    saveLog(f'num_anchors: {len(anchor_indices_list)}')

def load_node_features(name, data, G, share_data_folder):
    """Load node features from a file if available."""
    filepath = os.path.join(share_data_folder, 'nodeFeatM.npy')
    if os.path.exists(filepath):
        nodeFeatM = np.load(filepath)
        saveLog("load nodeFeatM")
        saveLog(f"shape of nodeFeatM: {nodeFeatM.shape}")
        return nodeFeatM
    else:
        nodeFeatM = generate_node_features(name, data, G, share_data_folder)
        return nodeFeatM

def generate_node_features(name, data, G, share_data_folder):
    """Generate node features for the dataset."""
    if name == "ogbn-proteins":
        nodeFeatM = generate_one_hot_features(data[name]['node_species'], G)
    else:
        nodeFeatM = data[name]['node_feat']
    np.save(os.path.join(share_data_folder, 'nodeFeatM.npy'), nodeFeatM)
    saveLog("create nodeFeatM")
    saveLog(f"shape of nodeFeatM: {nodeFeatM.shape}")
    return nodeFeatM

def generate_one_hot_features(node_species, G):
    """Generate one-hot encoded features based on node species."""
    nodeFeatList = np.unique(node_species)
    nodeFeatOneHotDict = {nodeFeat.item(): np.eye(len(nodeFeatList))[i] for i, nodeFeat in enumerate(nodeFeatList)}
    nodeFeatMatrix = np.array([nodeFeatOneHotDict[feat.item()] for feat in node_species])
    return nodeFeatMatrix

def load_or_generate_distance_matrices(name, data, shareDataFolder, anchorIndicesList):
    D_list_name = f"{shareDataFolder}/D_list_edge8.npy" if name == "ogbn-proteins" else f'{shareDataFolder}/D_list.npy'
    if os.path.exists(D_list_name):
        saveLog(f"Load D_list")
        D_list = [[]]*8 if name == "ogbn-proteins" else []
    else:
        D_list = create_distance_matrices(name, data, shareDataFolder, anchorIndicesList)
    return D_list

def create_distance_matrices(name, data, shareDataFolder, anchorIndicesList):
    D_list = []
    if name == "ogbn-proteins":
        for edge_feature_index in range(8):  # 8 features for "ogbn-proteins"
            D_path = f"{shareDataFolder}/D_list_edge{edge_feature_index}.npy"
            if not os.path.exists(D_path):
                saveLog(f"Creating D for edge feature {edge_feature_index}")
                D = process_edge_features_and_calculate_shortest_paths(data, name, edge_feature_index, anchorIndicesList)
                np.save(D_path, D, allow_pickle=True)
            else:
                saveLog(f"Exists: {D_path}, continue")
            D_list.append([])
    elif name == "ogbn-products":
        D_path = f"{shareDataFolder}/D_list_edge0.npy"
        if not os.path.exists(D_path):
            saveLog(f"Creating D for edge feature 0")
            D = process_edge_features_and_calculate_shortest_paths(data, name, 0, anchorIndicesList)
            np.save(D_path, D, allow_pickle=True)
        else:
            saveLog(f"Exists: {D_path}, continue")
        D_list.append([])
        
    return D_list

def process_edge_features_and_calculate_shortest_paths(data, name, edge_feature_index, anchorIndicesList):
    # Assuming 'data' contains edge indices and edge features under 'name'
    e1, e2 = data[name]['edge_index']
    n = data[name]['num_nodes']
    
    # Process edge features
    if name == "ogbn-proteins":
        edge_feats = data[name]['edge_feat'][:, edge_feature_index]
        # Adjust edge features as needed
        edge_feats[edge_feats == 1] = 1 - 1e-10
        # Create adjacency matrix
        A = sp.csr_matrix((1 - edge_feats, (e1, e2)), shape=(n, n))
    elif name == "ogbn-products":
        edge_feats = np.ones(e1.shape[0])
        A = sp.csr_matrix((edge_feats, (e1, e2)), shape=(n, n))
    
    

    # Calculate shortest paths
    num_anchors = len(anchorIndicesList)
    D = np.zeros((n, num_anchors))
    for j in range(num_anchors):
        anchor = anchorIndicesList[j]
        dist = sp.csgraph.shortest_path(A, indices=anchor, directed=False)
        D[:, j] = dist
    D[np.isinf(D)] = 0
    if (D > 1e6).any():
        saveLog("!!!there exist D > 1e6")
        D[D > 1e6] = 0

    #save D shape
    saveLog(f"create D shape: {D.shape}")
    
    return D

def normalize_transformed_data(UforWholeDisM):
    for i_col in range(UforWholeDisM.shape[1]):
        col_norm = np.linalg.norm(UforWholeDisM[:, i_col])
        UforWholeDisM[:, i_col] /= col_norm
    return UforWholeDisM


def apply_svd(D,numPCperGraph,name,component):
    if name in ["ogbn-products"]:
        D_forSVD = D[component, :]
    else:
        D_forSVD = D
        
    #save d svd shape
    saveLog(f"D_forSVD shape: {D_forSVD.shape}")
    U, Sigma, VT = np.linalg.svd(D_forSVD, full_matrices=False)
    if type(numPCperGraph) != int:
        SigmaSum = np.sum(Sigma**2)
        num_PCs = determine_num_components(Sigma, numPCperGraph, SigmaSum)
    else:
        num_PCs = numPCperGraph
    
    transformed_matrix = normalize_transformed_data(D @ VT.T)
    
    return transformed_matrix[:, :num_PCs]

def apply_DVCS(D, num_components):
    D_transformed = DVCS(D)
    D_transformed = normalize_transformed_data(D_transformed)
    return D_transformed[:, :num_components]

def determine_num_components(singular_values, numPCperGraph, SigmaSum):
    sSum = 0
    for j, s in enumerate(singular_values):
        sSum += s**2
        if sSum >= numPCperGraph * SigmaSum:
            return j + 1  # Number of components needed
    return len(singular_values)  # Fallback to use all components


def perform_gc_transformation(D_list, numPCperGraph, GCoption, shareDataFolder, name="ogbn-products", component=None):
    transformed_matrices = []
    for D_ith,D in enumerate(D_list):
        D = np.load(f'{shareDataFolder}/D_list_edge{D_ith}.npy')
        if GCoption == "TC":
            transformed_matrix = apply_svd(D,numPCperGraph,name,component)
        elif GCoption == "DVCS":
            transformed_matrix = apply_DVCS(D, numPCperGraph)
        else:
            raise ValueError(f"Unsupported GC option: {GCoption}")
        
        transformed_matrices.append(transformed_matrix)
    
    return transformed_matrices

def concatenate_matrices(list_of_matrices, nodeFeatM):
    concatenated_matrix = np.concatenate(list_of_matrices, axis=1)
    X = np.concatenate((concatenated_matrix, nodeFeatM), axis=1)
    return X


def save_final_matrix(X, shareDataFolder, GCoption, numPCperGraph, num_anchors, timestamp):
    if GCoption in ["TC", "DVCS"]:
        filename = f'{shareDataFolder}/X_numPCperGraph_{numPCperGraph}_GCoption_{GCoption}_numAnchors_{num_anchors}_Xshape{X.shape[0]}_{X.shape[1]}_timestamp{timestamp}.pt'
    else:
        filename = f'{shareDataFolder}/X.pt'
    # save size of X
    saveLog(f"X shape: {X.shape}")
    torch.save(torch.from_numpy(X), filename)
    return filename
