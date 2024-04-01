import json
from utils import *
from utils_train import *
import datetime
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split, Dataset
import argparse
import random
import os

# accept random seed from command line
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--random_seed', type=int, help='An integer for setting the random seed.')
parser.add_argument('--config_file', type=str, help='An str for setting the config file.', default='config.json')
args = parser.parse_args()

# Load the configuration from a JSON file
with open(args.config_file, "r") as json_file:
    config = json.load(json_file)

# Dynamically create variables in the global namespace
for key, value in config.items():
    globals()[key] = value

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
saveFolder = f'GCNN_{datasetName}_{timestamp}'
os.environ['SAVE_FOLDER'] = saveFolder
create_folder(saveFolder)
saveLog(timestamp)
random_seed = args.random_seed
saveLog(f"Random seed: {random_seed}")

if random_seed is None:
    random_seed = random.randint(0, 1000000)
    saveLog(f"Random seed: {random_seed}")




# print the configuration
for key, value in config.items():
    saveLog(f"{key}: {value}")

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # For multi-GPU setups.

shareDataFolder = f'shareData_{datasetName}'
create_folder(shareDataFolder)









data, label, dataset = load_datasets(datasetName)

if datasetName in ['ogbn-arxiv', 'ogbn-proteins']:
    torch.save(label[datasetName], f'{shareDataFolder}/label.pt')
else:
    torch.save(label[datasetName], f'{shareDataFolder}/label.pt')
    train_Y = to_one_hot(label[datasetName], int(torch.max(label[datasetName]).item())+1)
    torch.save(train_Y, f'{shareDataFolder}/train_Y.pt')
#log_dataset_info(data, label, datasetName)


G = initialize_graph(data, datasetName)

if datasetName in ['ogbn-products']:
    largest_component = analyze_components(G, shareDataFolder)
    handle_indices(dataset, shareDataFolder, largest_component)
else:
    largest_component = []
    handle_indices(dataset, shareDataFolder, largest_component)
del dataset

anchor_indices_list = load_or_generate_anchor_indices(shareDataFolder, largest_component, G, num_anchors, datasetName, data)

log_num_anchors(anchor_indices_list)

nodeFeatM = load_node_features(datasetName, data, G, shareDataFolder)

D_list = create_distance_matrices(datasetName, data, shareDataFolder, anchor_indices_list)

X_file_List = []   
for num_GCs in num_GCs_List:

    transformed_matrices = perform_gc_transformation(D_list, num_GCs, GCoption, shareDataFolder, name=datasetName, component=largest_component)
    X = concatenate_matrices(transformed_matrices, nodeFeatM)
    X_file_List.append(save_final_matrix(X, shareDataFolder, GCoption, num_GCs, num_anchors, timestamp))







device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
saveLog(f'Selected device: {device}')

# Define the transformations to be applied to the data
transform = transforms.Compose([
    transforms.ToTensor()
])



train_idx_file = f"{shareDataFolder}/train_idx_original.pt"
test_idx_file = f"{shareDataFolder}/test_idx_original.pt"
valid_idx_file = f"{shareDataFolder}/valid_idx_original.pt"
label_file = f"{shareDataFolder}/label.pt"
if datasetName in ['ogbn-products']:
    train_Y_file = f"{shareDataFolder}/train_Y.pt"
else:
    train_Y_file = label_file

label = torch.load(label_file)
y_shape = label.shape[1]
train_idx = torch.load(train_idx_file)
valid_idx = torch.load(valid_idx_file)
test_idx = torch.load(test_idx_file)

optimizer = torch.optim.Adam
#optimizer = torch.optim.RMSprop
if datasetName == "ogbn-products":
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCELoss()

evaluator = Evaluator(name = datasetName)

#hidden_dim = [[],[],[],[],[],[],[],[],[],[],[128,64],[128,64],[128,64],[128,64],[128,64],[128,64],[128,64],[128,64],[128,64],[128,64]]

for X_file in X_file_List:
    saveLog(f"X_file: {X_file}")
    # Load the training data using the train_loader
    train_dataset = MemoryMappedDataset(X_file, label_file,transform=None, device = device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    # Calculate the mean and standard deviation of the training data
    n_samples = 0
    mean = 0.0
    var = 0.0

    y_max = 0
    for X_train, y_train in train_loader:
        data = X_train
        batch_samples = data.size(0)
        x_shape = data[0].shape[0]
        data = data.view(batch_samples, -1)
        mean += data.mean(0)* batch_samples
        var += data.var(0) * batch_samples
        n_samples += batch_samples
        y_max = max(y_max, torch.max(y_train).item())
    y_max += 1
    
    if datasetName not in ['ogbn-proteins']:
        y_shape = int(y_max)

    mean /= n_samples
    var /= n_samples
    stddev = torch.sqrt(var)
    saveLog(f"mean: {mean}, std: {stddev}")

    # Define the transformations to be applied to the data
    transform = transforms.Compose([
        CustomNormalize(mean, stddev)
    ])
    
    del train_dataset, train_loader
    # create subsets of the data based on the indices
    memoryDataset = MemoryMappedDataset(X_file, label_file, device = device, transform=transform)
    
    if datasetName in ['ogbn-products']:
        memoryDatasetTrain = MemoryMappedDataset(X_file, train_Y_file, device = device, transform=transform)
        train_dataset = Subset(memoryDatasetTrain, train_idx)
    else:
        train_dataset = Subset(memoryDataset, train_idx)
    
    
    test_dataset = Subset(memoryDataset, test_idx)
    valid_dataset = Subset(memoryDataset, valid_idx)
    
    #save size of idx
    saveLog(f"train_idx: {len(train_idx)}")
    saveLog(f"test_idx: {len(test_idx)}")
    saveLog(f"valid_idx: {len(valid_idx)}")

    # create DataLoaders for the train, test, and validation sets

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    
    # save test results in a dictionary
    test_results = {}
    
    for h_dim in hidden_dim:
        
        #if h_dim is not a key in test_results
        if not tuple(h_dim) in test_results.keys():
            test_results[tuple(h_dim)] = []

        
        model = denseNet(input_dim=x_shape, hidden_dim=h_dim,out_dim=y_shape,datasetName=datasetName).to(device)
        saveLog(f"Model: {model}")
        saveLog(f"hidden_dim: {h_dim}")
        saveLog(f"numb of parameters: {sum(p.numel() for p in model.parameters())}")
        main_train_loop(model, train_loader, valid_loader, optimizer, criterion, evaluator, datasetName, num_epochs, earlyStopEpoch, saveFolder, device, lr)
        
        #load best model from f"{saveFolder}/best_model.pt"
        model.load_state_dict(torch.load(f"{saveFolder}/best_model.pt"))
        
        test_resulst = evaluate_model(model, test_loader, evaluator, datasetName, datasetType='test')
        saveLog(f"Test results: {test_resulst}")
        test_results[tuple(h_dim)].append(test_resulst)
    
    #save test results in a file
    saveLog(f"Test results: {test_results}")
    
    #save average with std of test results in a file, and the range of the test results
    test_results_avg = {}
    for key, value in test_results.items():
        test_results_avg[key] = {}
        test_results_avg[key]['avg'] = np.mean(value)
        test_results_avg[key]['std'] = np.std(value)
        saveLog(f"{key}: {test_results_avg[key]['avg']}±{test_results_avg[key]['std']}")

    