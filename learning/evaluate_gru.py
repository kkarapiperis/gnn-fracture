import sys
import time
import torch
import random
import argparse
import numpy as np
from model import *
from utilities_gru import *
from scipy.linalg import block_diag
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_dense_adj

def run():
    # Parser options
    parser = argparse.ArgumentParser()
    parser.add_argument('--Nx', type=int, default=13)
    parser.add_argument('--Ny', type=int, default=13)
    parser.add_argument('--shape', type=str, default='hex')
    parser.add_argument('--Nevals_train', type=int, default=10000)
    parser.add_argument('--Nevals_test', type=int, default=10000)
    parser.add_argument('--num_nodes', type=int, default=162)
    parser.add_argument('--notch_width', type=int, default=4)
    parser.add_argument('--data_augment', type=int, default=0)
    parser.add_argument('--normalize_input', type=int, default=1) 
    parser.add_argument('--encoder', type=str, default='GraphConvGRU')
    parser.add_argument('--decoder', type=str, default='InnerProduct')
    parser.add_argument('--coor_dim', type=int, default=2) 
    parser.add_argument('--edge_dim', type=int, default=3) 
    parser.add_argument('--node_dim', type=int, default=9)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--gnn_layers', type=int, default=6)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--min_disorder', type=float, default=0.1)
    parser.add_argument('--max_disorder', type=float, default=1.2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda') 
    parser.add_argument('--model_name', type=str, default='gru')
    args = parser.parse_args()

    # Training/Hyperparameters
    Nx = args.Nx
    Ny = args.Ny
    shape = args.shape
    Nevals_train = args.Nevals_train
    Nevals_test = args.Nevals_test
    num_nodes = args.num_nodes
    notch_width = args.notch_width
    normalize = args.normalize_input
    data_augment = args.data_augment
    coor_dim = args.coor_dim
    edge_dim = args.edge_dim
    node_dim = args.node_dim
    hidden_dim = args.hidden_dim
    gnn_layers = args.gnn_layers
    latent_dim = args.latent_dim
    min_disorder = args.min_disorder
    max_disorder = args.max_disorder
    dropout = args.dropout
    device = args.device
    model_name = args.model_name
    print(model_name)

    # Seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Output directory for saved model
    saveName = 'output/' + model_name

    # Load meshes which were used as train/test 
    train_mesh_idx = np.loadtxt(saveName + '-train_mesh_idx.dat').astype(int)
    test_mesh_idx = np.loadtxt(saveName + '-test_mesh_idx.dat').astype(int)
    train_mesh_idx = sorted(train_mesh_idx)
    test_mesh_idx = sorted(test_mesh_idx)

    # Batch vector which shows which holds the batch id for each node
    batch_size = 1
    batch = np.zeros(num_nodes*batch_size)
    for b in range(batch_size):
        batch[b*num_nodes:(b+1)*num_nodes] = b
    batch = torch.tensor(batch, device=device, dtype=torch.int64)

    # Find the maximum fracture sequence length
    max_sequence_length = getMaxSequenceLength(Nx, Ny, shape, train_mesh_idx)

    # Get train and test data loaders
    train_loaders,train_mesh_list = getDataLoaders(Nx, Ny, shape, notch_width, train_mesh_idx, normalize,
        data_augment, max_sequence_length, batch_size, shuffle=False)
    test_loaders,test_mesh_list = getDataLoaders(Nx, Ny, shape, notch_width, test_mesh_idx, normalize,
        0, max_sequence_length, batch_size, shuffle=False)

    # Read normalizing constants 
    x_bounds,_ = readNormalization(Nx, Ny, shape)

    # Define encoder
    if args.encoder == 'GraphConvGRU':
        encoder = GraphConvGRU(node_dim, hidden_dim, latent_dim, edge_dim, gnn_layers, dropout, batch)
    else:
        sys.exit('Unknown encoder type')

    # Define decoder
    if args.decoder == 'InnerProduct':
        decoder = InnerProductDecoder(batch_size=1)
    else:
        sys.exit('Unknown decoder type')

    # Define autoencoder 
    graph_model = GAEGRU(encoder, decoder)

    # Load trained model
    graph_model.load_state_dict(torch.load(saveName + '-model', map_location=torch.device(device)))

    # Move to device
    graph_model.to(device)
    graph_model.eval()

    # Save the fracture paths predicted from the beginning by repeatedly evaluating the model 
    edges_train_data = []
    edges_train_pred = []
    edges_test_data = []
    edges_test_pred = []
    
    # The broken edges forming the initial notch
    broken_edges_init = getNotchEdges(Nx, Ny, shape, notch_width)
        
    # Training
    mesh_idx = 0
    for data_all_steps in zip(*train_loaders):
        if mesh_idx >= Nevals_train: continue
        start = time.time()

        # Reset list of broken edges
        edges_data = broken_edges_init.copy()
        edges_pred = broken_edges_init.copy()

        # Store the true data
        true_sequence_length = 0
        for step in range(max_sequence_length):

            data = data_all_steps[step].to(device)

            # Stop if max step is reached
            if data.pad == 1 or step == max_sequence_length-1: 
                true_sequence_length = step
                break
                
            edge_idx_data = data.F_idx.detach().cpu().numpy()
            edges_data.append(list(edge_idx_data))

        edges_train_data.append(edges_data)

        # Do the prediction starting from intact mesh 
        data0 = data_all_steps[0].to(device)
        A0 = to_dense_adj(data0.edge_index).view(num_nodes,num_nodes) 
        x0 = data0.x; r = data0.edge_attr; M = data0.M
        A = A0.clone().detach()
        x = x0.clone().detach()
        edges_pred = broken_edges_init.copy()

        for step in range(max_sequence_length):        

            # Stop if max step is reached
            if step == true_sequence_length: 
                break

            # No hidden state for first step prediction
            if step == 0: 
                h = None

            # Predict the next edge to break
            edge_index = torch.nonzero(A).t().contiguous()
            h, F_pred = graph_model(x, edge_index, r, M, h)
            edge_idx_pred = torch.argmax(F_pred.view(1,-1),axis=1)
            edge_idx_pred = int(edge_idx_pred.detach().cpu().numpy())
            edges_pred.append(list(divmod(edge_idx_pred,num_nodes)))

            # Update the quantities for next prediction (adjacency matrix, etc)
            A, x, r, M = updateQuantitiesForNextStepPrediction(A0, A, x0, r, 
                x_bounds, edges_pred, device)

        edges_train_pred.append(edges_pred)

        # Write to file
        str_tmp = 'idx = ' + str(mesh_idx) + '/ ' + str(len(train_mesh_idx)) \
          + ', time = ' + str(time.time() - start) + '\n'

        with open(saveName + '-eval-progress-train.txt', 'a') as file_train:
            file_train.write(str_tmp)

        mesh_idx += 1

    f = open(saveName + '-edges_train_data.dat','w')
    for edge_list in edges_train_data:
        for edge in edge_list[notch_width-2:]:
            f.write('(%d,%d), ' % (edge[0], edge[1]))
        f.write('\n') 
    f.close()
    f = open(saveName + '-edges_train_pred.dat','w')
    for edge_list in edges_train_pred:
        for edge in edge_list[notch_width-2:]:
            f.write('(%d,%d), ' % (edge[0], edge[1]))
        f.write('\n') 
    f.close()
    file_train.close()

    # Same for testing 
    mesh_idx = 0
    for data_all_steps in zip(*test_loaders):
        if mesh_idx >= Nevals_test: continue
        start = time.time()

        # Reset list of broken edges
        edges_data = broken_edges_init.copy()
        edges_pred = broken_edges_init.copy()

        # Store the true data
        true_sequence_length = 0
        for step in range(max_sequence_length):

            data = data_all_steps[step].to(device)

            # Stop if max step is reached
            if data.pad == 1: 
                true_sequence_length = step
                break

            edge_idx_data = data.F_idx.detach().cpu().numpy()
            edges_data.append(list(edge_idx_data))

        edges_test_data.append(edges_data)

        # Do the prediction starting from intact mesh 
        data0 = data_all_steps[0].to(device)
        A0 = to_dense_adj(data0.edge_index).view(num_nodes,num_nodes) 
        x0 = data0.x; r = data0.edge_attr; M = data0.M
        A = A0.clone().detach()
        x = x0.clone().detach()
        edges_pred = broken_edges_init.copy()

        for step in range(max_sequence_length):        

            # Stop if max step is reached
            if step == true_sequence_length: 
                break

            # No hidden state for first step prediction
            if step == 0: 
                h = None

            # Predict the next edge to break
            edge_index = torch.nonzero(A).t().contiguous()
            h, F_pred = graph_model(x, edge_index, r, M, h)
            edge_idx_pred = torch.argmax(F_pred.view(1,-1),axis=1)
            edge_idx_pred = int(edge_idx_pred.detach().cpu().numpy())
            edges_pred.append(list(divmod(edge_idx_pred,num_nodes)))

            # Update the quantities for next prediction (adjacency matrix, etc)
            A, x, r, M = updateQuantitiesForNextStepPrediction(A0, A, x0, r, 
                x_bounds, edges_pred, device)

        edges_test_pred.append(edges_pred)

        # Write to file
        str_tmp = 'idx = ' + str(mesh_idx) + '/ ' + str(len(test_mesh_idx)) \
          + ', time = ' + str(time.time() - start) + '\n'

        with open(saveName + '-eval-progress-test.txt', 'a') as file_test:
            file_test.write(str_tmp)

        mesh_idx += 1

    f = open(saveName + '-edges_test_data.dat','w')
    for edge_list in edges_test_data:
        # Disregard the notch edges
        for edge in edge_list[notch_width-2:]:
            f.write('(%d,%d), ' % (edge[0], edge[1]))
        f.write('\n') 
    f.close()
    f = open(saveName + '-edges_test_pred.dat','w')
    for edge_list in edges_test_pred:
        # Disregard the notch edges
        for edge in edge_list[notch_width-2:]:
            f.write('(%d,%d), ' % (edge[0], edge[1]))
        f.write('\n') 
    f.close()
    file_test.close()

if __name__ == '__main__':
    run()