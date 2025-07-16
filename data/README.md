# DATA

For simplicity, we included most of the data (when suitable and not too large) directly in the repository to have everything in one place.

# Difference Graphs

We placed all model difference graphs in `diffgraphs_new.zip`. 
These include the different projects and their commit history.
We gained these graphs with the help of EMFCompare. 

# Embeddings

Unfortunately, the files of node embeddings are too large to be included all.  
However, they can easily be generated using `experiments/crosscutting_changes/preprocessing_add_node_embeddings.py`.
For quick testing, we included a very, very small subset in the folder. This subset is **not representative**, but gives a basic idea of the format etc.

# Network
The file `final_neural_network_model_epoch_3.pth` contains the trained neural network weights used in our evaluation. This model can be loaded directly for inference or further analysis.


# Evaluation

The final output of our baselines and our approach 'NextFocus' (i.e., the output of the "Testing" stage),  
which can be passed into the scripts under `experiments/crosscutting_changes/final_{}`, is provided in `output_ONLYONE4.zip`.
