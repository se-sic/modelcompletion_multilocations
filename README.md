# Multi-Location Software Model Completion

## Paper  

Additional information regarding the paper is given in the PDF file: `Multi-Location_Software Model_Completion.pdf` of the repository. More precisely, the paper has an appendix there; for more info, just scroll down to the appendix at the very end of the paper.

## Code 

Code to run is given in `experiments/crosscutting_changes`.

### Preprocessing 

To compute the embeddings, run the script `preprocessing_add_node_embeddings.py`,  
which will embed the nodes of the graphs. Please provide your OpenAI key there.

### Training 

Run `neural_network_analyser.py` for training the neural network.

Run `neural_network_hyperparametertuning.py` for hyperparameter tuning.

Run `historical_only_analyser.py`, which builds the adjacency matrix for the historical baseline from the train set.

### Testing 

All scripts start with `test_eval`.  
The testing scripts for the baseline semantics, historical, random, and neural network approach "NextFocus" are named `test_eval_{approach}`.

These will be passed through the network.

### Final statistics and eval 

Start with `final_`:

- `final_comparison.py` for statistics on Experiment 1 and Experiment 3  
- `final_eval_graph_radius.py` for Experiment 2  
- `final_eval_dataset_size.py` for Experiment 3  

### Additional info

Files starting with `Helper` are called by the rest.  
Modules follow the same pattern.

## Data

The data and intermediate results of the approach are for simplicity provided in the `data` folder.
