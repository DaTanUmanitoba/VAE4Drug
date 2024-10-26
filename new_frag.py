import argparse
import os
import json

import torch
from collections import Counter
import json

def load_model(path):
    model_dict = torch.load(path)
    model = model_dict['type'](model_dict['args']).cuda()
    model.load_state_dict(model_dict['model'])
    return model

def calculate_accuracy(logits, true_labels):
    """
    Calculate accuracy given logits and true labels.
    
    """
    predictions = torch.argmax(logits, dim=1)
    correct_predictions = torch.eq(predictions, true_labels).float()  # Convert to float for division
    accuracy = correct_predictions.sum() / len(true_labels)
    
    return accuracy.item() 

# remove duplicates
def count_duplicates(lst):
    element_count = Counter(lst)
    duplicates = sum((count - 1) for element, count in element_count.items() if count > 1)
    return duplicates

def remove_duplicates_and_save(lst, filename):
    seen = set()
    unique_items = [x for x in lst if x not in seen and not seen.add(x)]
    
    with open(filename, 'w') as f:
        json.dump(unique_items, f, indent=4)
    
    return unique_items
 
    
def main():
    parser = argparse.ArgumentParser(description="Conditionally sample and decode molecule fragments from the latent space")
    parser.add_argument('--model_path', type=str, required=True, help='path of the VQVAE model from which the fragments are sampled')
    parser.add_argument('--output_dir', type=str, required=True, help='output dir for the synthesized fragments')
    parser.add_argument('--batch_size', type=int, default=50, help='The number of fragments to be generated')
    parser.add_argument('--num_embed', type=int, default=10, help='number of embedding vectors in the codebook')
    parser.add_argument('--latent_dim', type=int, default=10, help='The dimension of the latent vector')
    parser.add_argument('--pred_threshold', type=float, default=0.5, help='threshold above which the new fragments will be selected')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    out_file = args.output_dir + '/vqVAE_sample_frags.json'

    model = load_model(args.model_path).cuda()
    model.eval()

    # Randomly sample indices within the learned embedding space
    random_indices = torch.randint(low=0, high=args.num_embed, size=(args.batch_size, args.latent_dim, 1)).cuda()
    decoded_data = model.decode_action(random_indices, args.pred_threshold)

    num_duplicates = count_duplicates(decoded_data)
    print(f"Number of duplicates in the list: {num_duplicates}")
    resulting_list = remove_duplicates_and_save(decoded_data, out_file)

    print(f"Length of resulting list without duplicates: {len(resulting_list)}")
    print(f'The first ten synthesized fragments: {resulting_list[1:10]}') 
    

if __name__ == '__main__':
    main()
