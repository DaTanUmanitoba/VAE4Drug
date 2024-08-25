## Part of the implementation is inspired by and based on the work of: 
https://github.com/benatorc/fast_explore/ (Mostly the part of the Graph embedding for molecules )

## Folder structure:
1. ./data: all the experiment datasets tried in this study, inculding GSK3, JNK3 and BACE datasets
2. ./model: The VQ-VAE model implementation, including ./model/mol_graph, the graph embedding initialization for the molecules
3. ./output: The generated data and results
4. ./stat_analysis: some statistical results for the benchmark datasets
5. ./*py: pipeline scripts, see contents below
6. ./*txt: the training trajectory files

## Enviroment: 
I work with Python 3.8, but later versions should be Okay.

Install packages listed in ./requirements.txt

## How to run:
For each of the dataset, the analysis pipeline is summed into a '.sh' file. For example the 'jnk3.sh' is for the dataset 'JNK3'

Description of the scripts: (see the example in "jnk3.sh" and help information for each script)

Suppose you have your training data in the folder "./data/" (example: ./data/gsk3/training.csv)

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

python latent_sampling.py 

### Step 6:
Merge the synthesized fragments with the backbones to create new molecules

python sample_merge_molecules.py 

### Step 7:
Compute the metrics analysis for the created molecules

python eval_metrics.py 

### The final result is in the folder of ./output/output_{dataset}/, it contains:
1) models/: the trained models
2) generated_mols/: The generated molecules
3) eval_result/: The evaluation results and statistics
4) Some more intemediate results in this folder.