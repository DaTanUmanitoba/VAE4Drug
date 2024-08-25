## for the gsk3 data

dir=`pwd`
label="gsk3"

out_dir=${dir}/output/output_${label}/

echo "The working data path: " ${dir}/data/
echo "The output path: " $out_dir

################################################################
#Suppose you have your training data in the folder:  ${dir}/data/${label}/training.csv

##train a rf classifier as the oracle predictor
# python rf_train.py --data ${dir}/data/${label}/training.csv \
#                    --output_dir ${dir}/data/${label}


##ANALYSIS PIPELINE 
################################################################
##genereate a set of side chains from the input molecules
# python create_side_chain.py --data_path ${dir}/data/${label}/training.csv \
#                                 --output_dir ${dir}/data/${label}/


# # create a SELFIES string vocabulary for the side chains
# python create_selfie_vocab.py   --data_path ${dir}/data/${label}/fragments.csv \
#                                 --output_path ${dir}/data/${label}/vocab_selfies_sanitized.plk \



# Train the guided-VQ-VEQ model
# python train.py --data_path ${dir}/data/${label}/fragments.csv\
#                 --selfies_vocab ${dir}/data/${label}/vocab_selfies_sanitized.plk\
#                 --output_path ${out_dir} \
#                 --batch_size 32 --latent_dim 12 --num_embed 12 --agg_depth 4 --rnn_depth 4 \
#                 --hidden_dim 256  --vq_coef 1. --commit_coef 1. --pred_coef 1. --recon_coef 1. \
#                 --epochs 300 --train_ratio 0.9 



# Generate synthesized fragments from the trained model
python latent_sampling.py --model_path  ${out_dir}/models/model_best \
                         --output_dir ${out_dir}\
                        --batch_size 3000 --latent_dim 12 --num_embed 12 \
                        --pred_threshold 0.6



# Merge the synthesized fragments with the backbones to create new molecules
python sample_merge_molecules.py --filtered_frag_path ${dir}/data/${label}/fragments.csv \
                                --vqvae_frags  ${out_dir}/vqVAE_sample_frags.json \
                                --output_dir ${out_dir}/generated_mols \
                                --n_samples 2



# Compute the metrics analysis for the created molecules
python eval_metrics.py --generated_data ${out_dir}/generated_mols/merged_molecules.csv \
                        --output_path ${out_dir}/eval_result \
                        --novelty_threshold 0.6 --dataset ${label} \
                        --classifier_dir ${dir}/data/${label}/rf_model.pkl