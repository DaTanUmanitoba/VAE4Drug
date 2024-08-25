import argparse
import selfies as sf
import pickle
import csv

def create_selfie_vocabulary(smiles_list, output_file_path):
    """
    Generate a vocabulary dictionary from a list of molecules in SMILES format, translate them
    into SELFIES format and encode each unique SELFIE symbol to an index.
    Adds a special padding symbol '[nop]' to the vocabulary.
    """
    selfie_dataset = []
    for smiles in smiles_list:
        try:
            selfie_code = sf.encoder(smiles)
            if selfie_code:
                selfie_dataset.append(selfie_code)
            else:
                print('Warning: Encoded SELFIE is None!')
        except Exception as error:
            print(f'SELFIE encoding error: {error}')
            
    print(f'Encoded SELFIE count: {len(selfie_dataset)}')
    vocabulary = sf.get_alphabet_from_selfies(selfie_dataset)
    vocabulary.add('[nop]')
    vocabulary = sorted(vocabulary)
    print(f'Vocabulary: {vocabulary}')  

    symbol_index_map = {symbol: idx for idx, symbol in enumerate(vocabulary)}
    print(symbol_index_map)
    
    with open(output_file_path, 'wb') as file:
        pickle.dump(symbol_index_map, file)

def process_input_data(input_file_path, output_file_path):
    """
    Process the input CSV to extract SMILES strings and generate a SELFIE vocabulary.
    """
    smiles_list = []
    with open(input_file_path, newline='') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            smiles_list.append(row[2])
    
    print(f'Sample SMILE from input: {smiles_list[2]}')
    create_selfie_vocabulary(smiles_list, output_file_path)

def main():
    parser = argparse.ArgumentParser(description='Generate SELFIE vocabulary from SMILES format of molecules.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file with SMILES strings.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output vocabulary.')

    args = parser.parse_args()
    process_input_data(args.data_path, args.output_path)

if __name__ == '__main__':
    main()
