import argparse
import os
import json
import csv
import random
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model.data import FragmentData
from model.vqvae import VQVAE
from model.mol_graph.vocab import atom_vocabulary


def load_json_file(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def load_csv_file(filepath):
    with open(filepath, newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        return list(reader)

def compute_accuracy(predictions, labels):
    correct = (predictions.argmax(dim=1) == labels).float()
    return correct.mean().item()

def setup_directories(output_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'models'), exist_ok=True)

# Loads a SELFIES vocabulary from a pickle file.
def load_vocab(filepath):
    with open(filepath, 'rb') as file:
        return pkl.load(file)

def create_dataset(data, atom_vocab, selfies_indx, batch_size):
    side_chain = [x[2] for x in data]
    labels = [float(x[3]) for x in data]
    return FragmentData(side_chain, labels, atom_vocab, selfies_indx, batch_size)

def save_model(path, model, args):
    model_dict = {'model': model.state_dict(), 'type': VQVAE, 'args': args}
    torch.save(model_dict, path)
    
def load_model(path):
    model_dict = torch.load(path)
    model = model_dict['type'](model_dict['args']).cuda()
    model.load_state_dict(model_dict['model'])
    return model

# Main function: organize the training process
def train_model(args):
    setup_directories(args.output_path)
    data = load_csv_file(args.data_path)
    random.seed(42)
    random.shuffle(data)
    split_index = int(args.train_ratio * len(data))
    train_data, eval_data = data[:split_index], data[split_index:]
    
    args.selfies_vocab = load_vocab(args.selfies_vocab)
    print(f'selfies: {args.selfies_vocab}')

    model_cls = VQVAE
    model = model_cls(args).cuda()

    train_dataset = create_dataset(train_data, args.atom_vocab, args.selfies_vocab, args.batch_size)
    eval_dataset = create_dataset(eval_data, args.atom_vocab, args.selfies_vocab, args.batch_size)
    
    #Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    #Xaxier weight initialization
    for param in model.parameters():
        if param.dim() != 1:
            nn.init.xavier_normal_(param)
        else:
            nn.init.constant_(param, 0)
          
    train(args, model, train_dataset, eval_dataset, optimizer)

    
# Handles the training for each epoch and evaluates the model.
def train(args, model, train_dataset, eval_dataset, optimizer):
    batch_stats = None
    highest_loss = float('inf')
    best_model_epoch = 0
    
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_dataset):
            loss, batch_stats, selfies_preds, class_logits, selfies_orig, labels = model.forward(*batch)
            accuracy = compute_accuracy(class_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
        print(f"Epoch {epoch}: Training: " + ", ".join(f"{key}: {value:.4f}" for key, value in batch_stats.items()) + f"; predictor accuracy: {accuracy:.4f}" )
        eval_loss = evaluate(model, eval_dataset, epoch)
        
        if eval_loss < highest_loss:
            best_model_epoch = epoch
            
    #save to the best model if the eval loss is the smallest   
    best_model = args.output_path + '/models/model_' + str(best_model_epoch)
    print(f'Training finished, saving the best model:  {best_model}')
    model = load_model(best_model)
    best_path = args.output_path + '/models/model_best'
    save_model(best_path, model, args)
    
# Evaluates the model on the test dataset.
def evaluate(model, dataset, epoch):
    model.eval()
    total_loss, recon_loss, vq_loss, commit_loss, prediction_loss, total_accuracy = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for batch in dataset:
            loss, batch_stats, selfies_preds, class_logits, selfies_orig, labels = model.forward(*batch)
            accuracy = compute_accuracy(class_logits, labels)
            total_loss += loss
            recon_loss += batch_stats['recon_loss']
            vq_loss += batch_stats['vq_loss']
            commit_loss += batch_stats['commit_loss']
            prediction_loss += batch_stats['prediction_loss']
            total_accuracy += accuracy
        avg_loss = total_loss / len(dataset)
        avg_accuracy = total_accuracy / len(dataset)
        avg_recon = recon_loss / len(dataset)
        avg_vq = vq_loss / len(dataset)
        avg_commit = commit_loss / len(dataset)
        avg_pred = prediction_loss / len(dataset)
        print(f'Epoch {epoch}: Eval: Total loss {avg_loss:.4f}, Recon loss {avg_recon:.4f}, Vq loss {avg_vq:.4f}, Commitment loss {avg_commit:4f}, prediction loss {avg_pred:.4f}, Accuracy {avg_accuracy:.4f}')

        #save model for each epoch
        model_path = args.output_path + '/models/model_' + str(epoch)
        save_model(model_path, model, args)
    
    return avg_loss    
        

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Data path containing the training dataset "fragments.csv" ')
    parser.add_argument('--output_path', type=str, required=True, help='path for the outpput files')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='train/test ratio')

    parser.add_argument('--selfies_vocab', type=str, required=True, help='The molecule string alphabete in SELFIES format')
    parser.add_argument('--atom_vocab', type=str, default= atom_vocabulary, help='The atom alphabete in one-hot encoding index')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size for the training')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension for the encoder and decoder output/input ')
    parser.add_argument('--latent_dim', type=int, default=10, help='the dimension of the latent vector, need to match with the number of embedding vectors in the codebook')
    parser.add_argument('--num_embed', type=int, default=10, help ='number of embeddings for the categorical codebook')
    parser.add_argument('--agg_depth', type=int, default=4, help='number of steps for the aggregation operation of the encoder')
    parser.add_argument('--rnn_depth', type=int, default=4, help='number of repeats for th RNN unit of the decoder')

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--grad_norm', type=float, default=10.0, help='Threshold for the gradient norm clipping')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs for training')
    
    parser.add_argument('--recon_coef', type=float, default=1.0, help='weight for the reconstruction loss')
    parser.add_argument('--vq_coef', type=float, default=1.0, help='weight of the loss for the vector quantization')
    parser.add_argument('--commit_coef', type=float, default=1.0, help='weight of the committment loss')
    parser.add_argument('--pred_coef', type=float, default=1.0, help='weight of the prediction loss for the binary classifier')
        
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    train_model(args)
