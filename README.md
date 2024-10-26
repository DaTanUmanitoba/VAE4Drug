## Folder structure:
1. ./data: query and download datasets here, inculding GSK3, JNK3 and BACE datasets
2. ./model: The VQ-VAE model implementation, including ./model/mol_graph, the graph embedding initialization for the molecules
3. ./*py: pipeline scripts, see contents below

## Enviroment: 
I work with Python 3.8, but later versions should be Okay.

Install packages listed in ./requirements.txt

## How to run:
Suppose you have downloaded your training data in the folder "./data/" (example: ./data/gsk3/training.csv)

### Step 1:
train a randomforest classifier based on Morgan fingerprints of the molecules as the oracle predictor for the gsk3 and jnk3 dataset, 
for BACE dataset, the oracle predictor is trained using chemprop package becuase the rf model performs bad.

python rf_train.py

### Step 2:
genereate a set of side chains from the input molecules

python create_side_chain.py 

### Step 3:
create a SELFIES string vocabulary for the side chains

python create_selfie_vocab.py  

### Step 4:
train the guided-VQ-VEQ model

python train.py 

### Step 5:
Generate synthesized fragments from the trained model

python new_frag.py 

### Step 6:
Merge the synthesized fragments with the backbones to create new molecules

python merge_molecules.py 

### Step 7:
Compute the metrics analysis for the created molecules

python eval_metrics.py 

### The final model and results will be in the folder of ./output/output_{dataset}/, it contains:
