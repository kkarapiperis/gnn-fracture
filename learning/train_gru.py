import sys
import time
import torch
import random
import argparse
import numpy as np
from model import *
from utilities_gru import *
from torch_geometric.data import Data, DataLoader

def run():
    # Parser options
    parser = argparse.ArgumentParser()
    parser.add_argument('--Nx', type=int, default=13)
    parser.add_argument('--Ny', type=int, default=13)
    parser.add_argument('--shape', type=str, default='hex')
    parser.add_argument('--Ndata', type=int, default=60000)
    parser.add_argument('--num_nodes', type=int, default=162)
    parser.add_argument('--notch_width', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--data_augment', type=int, default=1)
    parser.add_argument('--normalize_input', type=int, default=1) 
    parser.add_argument('--test_size', type=float, default=0.1)
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
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=6) 
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda') 
    parser.add_argument('--scheduler', type=str, default='StepLR') 
    parser.add_argument('--optimizer', type=str, default='Adam')  
    parser.add_argument('--model_name', type=str, default='gru')
    args = parser.parse_args()

    # Training/Hyperparameters
    Nx = args.Nx
    Ny = args.Ny
    shape = args.shape
    Ndata = args.Ndata
    epochs = args.epochs 
    num_nodes = args.num_nodes
    notch_width = args.notch_width
    batch_size = args.batch_size
    normalize = args.normalize_input
    data_augment = args.data_augment
    test_size = args.test_size
    coor_dim = args.coor_dim
    edge_dim = args.edge_dim
    node_dim = args.node_dim
    hidden_dim = args.hidden_dim
    gnn_layers = args.gnn_layers
    latent_dim = args.latent_dim
    min_disorder = args.min_disorder
    max_disorder = args.max_disorder
    dropout = args.dropout
    learning_rate = args.learning_rate 
    device = args.device
    patience = args.patience
    weight_decay = args.weight_decay
    model_name = args.model_name

    # Split to train/test data
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Split to train, test data
    train_mesh_idx, test_mesh_idx = splitTrainTestMeshes(Nx, Ny, shape, Ndata, min_disorder,
        max_disorder, test_size)

    # Find the maximum fracture sequence length 
    max_sequence_length = getMaxSequenceLength(Nx, Ny, shape, train_mesh_idx)

    # Output directory
    saveName = 'output/' + model_name
    np.savetxt(saveName + '-train_mesh_idx.dat', train_mesh_idx, fmt='%d')
    np.savetxt(saveName + '-test_mesh_idx.dat', test_mesh_idx, fmt='%d')

    # Get train and test data loaders
    train_loaders,_ = getDataLoaders(Nx, Ny, shape, notch_width, train_mesh_idx, normalize,
        data_augment, max_sequence_length, batch_size, shuffle=True)
    test_loaders,_ = getDataLoaders(Nx, Ny, shape, notch_width, test_mesh_idx, normalize,
        0, max_sequence_length, batch_size, shuffle=True)

    data_sets_train = len(train_loaders) * len(train_loaders[0].dataset)
    data_sets_test = max(1,len(test_loaders) * len(test_loaders[0].dataset))

    # Batch vector which shows which holds the batch id for each node
    batch = np.zeros(num_nodes*batch_size)
    for b in range(batch_size):
        batch[b*num_nodes:(b+1)*num_nodes] = b
    batch = torch.tensor(batch, device=device, dtype=torch.int64)
    num_batches = len(train_loaders[0].dataset)/batch_size
    
    # Define encoder
    if args.encoder == 'GraphConvGRU':
        encoder = GraphConvGRU(node_dim, hidden_dim, latent_dim, edge_dim, gnn_layers, dropout, batch)
    else:
        sys.exit('Unknown encoder type')

    # Define decoder
    if args.decoder == 'InnerProduct':
        decoder = InnerProductDecoder(batch_size)
    else:
        sys.exit('Unknown decoder type')

    # Define autoencoder 
    graph_model = GAEGRU(encoder, decoder)

    # Move to device
    graph_model.to(device)

    # Define optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(graph_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    torch.autograd.set_detect_anomaly(True)

    # Define scheduler
    if args.scheduler == 'ConstantLR':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs) 
    elif args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9) 
    else:
        sys.exit('Unknown scheduler option')

    loss_curve = []
    test_loss_curve = []
    startTotal = time.time()

    # Early stopping
    last_loss = 1e6
    trigger_times = 0

    # Define testing
    def test(graph_model, test_loaders):

        # Switch to evaluation mode
        graph_model.eval()
        total_loss = 0.
        data_sets_test = 0

        with torch.no_grad():
            # For each batch of meshes
            for data_all_steps in zip(*test_loaders):

                # Forward pass for each timestep
                for step in range(max_sequence_length):

                    # No hidden state for first step prediction
                    if step == 0: 
                        h = None

                    data = data_all_steps[step].to(device)
                    mask = torch.block_diag(*torch.split(data.M,num_nodes)) 
                    data_F = torch.zeros_like(mask, dtype=torch.float32)
                    for b in range(batch_size):
                        data_F[data.F_idx[2*b]+b*num_nodes,
                               data.F_idx[2*b+1]+b*num_nodes] = 1
                        data_F[data.F_idx[2*b+1]+b*num_nodes,
                               data.F_idx[2*b]+b*num_nodes] = 1
                    h, F_pred = graph_model(data.x, data.edge_index, 
                        data.edge_attr, mask, h)
                    pad_mask = torch.ones_like(mask, dtype=torch.float32)
                    for b in range(batch_size):
                        if data.pad[b]: 
                            pad_mask[b*num_nodes:(b+1)*num_nodes,
                                     b*num_nodes:(b+1)*num_nodes] = 0
                    loss = BCELoss(2*F_pred, data_F, pad_mask)
                    total_loss += loss.item()

                    for b in range(batch_size):
                        if data.pad[b] == 0:
                            data_sets_test += 1

        data_sets_test = max(data_sets_test, 1)
        return total_loss/data_sets_test

    # Main training loop
    for epoch in range(epochs):

        start = time.time()

        # Switch to training mode
        graph_model.train()

        # Initialize accumulated loss
        total_loss = 0.
        data_sets_train = 0

        # For each batch of meshes
        batch_idx = 0
        for data_all_steps in zip(*train_loaders):

            # loss = 0
            batch_idx += 1

            # Forward pass for each timestep
            for step in range(max_sequence_length):

                # No hidden state for first step prediction
                if step == 0: 
                    h = None

                data = data_all_steps[step].to(device)
                mask = torch.block_diag(*torch.split(data.M,num_nodes))  
                data_F = torch.zeros_like(mask, dtype=torch.float32)
                for b in range(batch_size):
                    data_F[data.F_idx[2*b]+b*num_nodes,
                           data.F_idx[2*b+1]+b*num_nodes] = 1
                    data_F[data.F_idx[2*b+1]+b*num_nodes,
                           data.F_idx[2*b]+b*num_nodes] = 1
                if step == 0:
                    h, F_pred = graph_model(data.x, data.edge_index, 
                        data.edge_attr, mask, h)
                else:
                    h, F_pred = graph_model(data.x, data.edge_index, 
                        data.edge_attr, mask, h.detach())
                pad_mask = torch.ones_like(mask, dtype=torch.float32)
                for b in range(batch_size):
                    if data.pad[b]: 
                        pad_mask[b*num_nodes:(b+1)*num_nodes,
                                 b*num_nodes:(b+1)*num_nodes] = 0
                loss = BCELoss(2*F_pred, data_F, pad_mask)
                
                for b in range(batch_size):
                    if data.pad[b] == 0:
                        data_sets_train += 1

                # Compute gradient after accumulating loss from the whole sequence 
                loss.backward()
                        
                # Update the weights using the gradient
                optimizer.step()
                optimizer.zero_grad()

                # Keep accumulated loss during epoch
                total_loss += loss.item()  

        total_loss /= data_sets_train

        # Test loss
        test_loss = test(graph_model, test_loaders)
        loss_curve.append(total_loss)
        test_loss_curve.append(test_loss)

        # Early stopping
        if test_loss > last_loss:
            trigger_times += 1
            if trigger_times > patience:
                print('Trigger Times:', trigger_times)
                print('Overfitting.. Stopping the training')
                break
        else:
            trigger_times = 0

        # Stop if things blow up
        if total_loss > 1000:
            print('Blow up - Total loss = ', total_loss)
            break
            
        last_loss = test_loss

        # Update scheduler
        scheduler.step()

        # Print statistics
        print("Epoch:", '%04d' % (epoch + 1) \
          ,", loss = ", "{:.8f}".format(total_loss)\
          ,", test loss = ", "{:.2f}".format(test_loss)\
         )
        sys.stdout.flush()

        # Write to file
        str_tmp = 'Epoch: ' + str(epoch + 1) \
          + ', loss = ' + str(round(total_loss, 4)) \
          + ', test loss = ' + str(round(test_loss , 4)) \
          + ', time = ' + str(time.time() - start) \
          + '\n'

        with open(saveName + '-loss.txt', 'a') as the_file:
            the_file.write(str_tmp)
            
    # Save the trained model
    torch.save(graph_model.state_dict(),saveName + '-model')
    torch.save(optimizer.state_dict(),saveName + '-optimizer')

    # Total time
    print('Total time (s):', time.time()-startTotal)
    the_file.close()
    graph_model.eval()

    # Save loss
    np.savetxt(saveName + '-loss.csv', np.c_[loss_curve, test_loss_curve], 
        delimiter=",", header="Train loss, Test loss")

if __name__ == '__main__':
    run()