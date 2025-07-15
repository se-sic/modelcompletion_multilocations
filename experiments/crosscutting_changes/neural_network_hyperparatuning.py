import itertools
import os
import sys
import optuna
import torch
from datetime import datetime
from torch.utils.data import DataLoader

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

# Import project modules
from experiments.crosscutting_changes.HELPER_configuration import COMBINATION
from experiments.crosscutting_changes.HELPER_neural_networks_dataloader import create_datasets
from experiments.crosscutting_changes.neural_network_analyser import SAMPLE_SUBSET, train_model
from experiments.crosscutting_changes.HELPER_neural_networks import AttentionClassifier, NonLinearModel

# Device setup
device = torch.device('cuda' if torch.cuda.is_available()
                     #  else 'mps' if torch.backends.mps.is_available()
                         else 'cpu')



#input_path = "/scratch/welter/CrossCutting/dataset_node_embeddings_text-embedding-3-small-with-ids/diffgraphs/"  
     
#directory_path_data =  "/scratch/welter/CrossCutting/neuralnetworkpairs_gpu/embedding_data_new_branch_ONLYONE_mergedemb/"
  
directory_path_data =   "../output_dataset_label/hyperparametertuning_trainset" #previously EQUAL to  neuralnetworkpairs_gpu/data_input_nn_noprespres/"
input_path = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids_small/diffgraphs/"
#input_path = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids-indivual-embeddings_small/diffgraphs/"


#for output
base_path = "../output_dataset_label/neural_network_data_small_output/"
  #base_path =  "/scratch/welter/CrossCutting/outputneuralnetwork/ZZ_hyperparametertuning_BASIC/"

EPOCHS=2

TRIALS=200



# Objective function for Optuna
def objective(trial, train_dataset, val_dataset, input_dim, networktype):
   
    # ───── non‑linear MLP branch ────────────────────────────────────────────
    if networktype=="nonlinear": 

        batch_size=256
        lr = 1e-3 #trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        #The analysis of the full file confirms that while the logarithmic search over 1e-4 to 1e-1 is important to cover a wide range of possibilities, the best performance (as indicated by significant improvements in top‑5 precision and robust test accuracy) is achieved with learning rates around 1e-3. Higher learning rates (above approximately 1e-2) tend to lead to either rapid but unproductive convergence (often stagnating at baseline metrics) or instability, while too low a rate would slow learning excessively.

        #[2048, 128]
        # Generate powers of 2 from 64 up to 16384
        powers_of_2 = [2 ** i for i in range(6, 15)]  # 2^6 = 64, 2^14 = 16384

        # Generate all valid (larger, smaller) combinations
        valid_combinations = [
            [a, b] for a, b in itertools.product(powers_of_2, powers_of_2)
            if a > b
        ]
        
        hidden_dims = trial.suggest_categorical("hidden_dims", valid_combinations)   

        # When no hidden layers are used ([]) or when only very small hidden dimensions are provided (such as [4] or even [16] or [4, 4]), the network almost always converges to a trivial solution.                                        
        #[], [4], [16], [4, 4], [4, 16],

        # Trials using a single hidden layer with a moderate number of neurons (for example, [128]) show a marked improvement. In these runs, the top‑5 precision steadily improved over epochs, often reaching values in the range of 0.55–0.56.
        #  In contrast, using [2028]—a much larger single-layer architecture—can sometimes deliver a similar improvement if the remaining hyperparameters (like learning rate and loss function settings) are tuned correctly. However, the very high dimensionality also raises a risk of overfitting or numerical instability if not carefully controlled.
        # Thus, while a larger hidden size like 2028 might provide more expressive power, a moderate value (around 128) appears to be a more robust candidate, trading off capacity with ease of optimization.
        # Configurations with two or three layers show the best potential
        # test 1 only  [128],[2024], [4048], 
        #That said, some choices like [16, 2] or [4, 16] may be on the lower end in terms of capacity, and the log indicates that those configurations sometimes fail to break out of the baseline performance, likely because the effective network capacity remains too constrained to capture complex decision boundaries.
        #In trials using [128, 64], the network had a sufficiently rich representation while keeping the parameter count moderate. This configuration generally showed a smooth and steady progression in top‑5 precision that climbed into the high 50-percentile range without signs of instability. The slightly “narrower” second layer appears to balance the network capacity with robustness, avoiding overfitting while still capturing complex patterns.

        # On the other hand, [128, 128] adds extra capacity by duplicating the first hidden layer’s size in the second layer. In the logs, networks using [128, 128] sometimes reached comparable top‑5 precision values—in certain runs even slightly higher—but this configuration can also be more sensitive to the overall tuning of the learning rate and loss parameters. The higher parameter count may lead to marginal improvements in representation power, but it can also increase the risk of overfitting if the remaining hyperparameters are not recalibrated.
        # [128, 64], [4048, 128],  [8096, 128], [8096, 2024], [4048, 1024],
        #[256, 128, 32][64, 8, 64][16, 128, 16])
        model = NonLinearModel(input_dim=input_dim, hidden_sizes=hidden_dims).to(device)
        print(f"[{datetime.now()}] Trial {trial.number}: "
              f"lr={lr}, batch_size={batch_size}, hidden_dims={hidden_dims}")    
    
    # ───── plain attention branch ───────────────────────────────────────────
    elif (networktype=="attention"):  
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 3, 4, 6, 8, 12, 16])
        batch_size = 1024
    
    
        # 2) Suggest dropout rate
        dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        model = AttentionClassifier(embed_dim=1536, num_heads=num_heads, dropout=dropout).to(device)
     
        print(f"[{datetime.now()}] Trial {trial.number}: "
            f"lr={lr}, "
            f"batch_size={batch_size}, "
            f"num_heads={num_heads}, "
            f"dropout={dropout}, ")


    #################################################################
    #############LOSS################################################
    #################################################################
    #Low pos_weight_factor (around 0.5–1.0) When the pos_weight_factor is set very low (for example, near 0.9 as seen in some trials), the network does not put enough extra emphasis on the minority class.
    #This suggests that the imbalance is not being effectively corrected, and the loss function does not “focus” the learning process on misclassified samples from the underrepresented class.
    # When the pos_weight_factor is pushed toward the upper end of the search range (for instance, values around 4.7–4.8 or even 4.0 in some trials), the logs show problems. In one scenario, extremely high values led to training instability—sometimes the loss even became “nan,” or the model’s metrics stalled at the baseline level.
    #pos_weight_factor = trial.suggest_float('pos_weight_factor', 0.5, 5.0, step=0.1)
    pos_weight_factor = trial.suggest_float('pos_weight_factor', 1.0, 4.0, step=0.1)
    #When focal_loss_alpha is set in the higher end of the range (between 0.7 and 1.0), as seen in Trials 6, 7, 11, 12, 13, 14, 17, 20, 23, and 24, the performance improves notably.
    # For instance, Trials 7, 11, and 12 (using α of 1.0) as well as those with values near 0.8 to 0.9 report progressive increases in top‐5 precision that can reach into the mid-50 percentiles.
    #Trial comparisons indicate that while some improvement in top‑5 precision occurs when moving into the intermediate range, the most substantial gains appear once α is pushed above roughly 0.7
    focal_loss_alpha = trial.suggest_float('focal_loss_alpha', 0.7, 1.0, step=0.1)  # Class balance parameter
    
    #• In Trial 4, where γ was set to 0.0, the model’s top‑5 precision remained stuck at a baseline value (around 0.0959) throughout training. In this setting, the focal loss does not distinguish between easy and hard samples; it essentially functions like standard cross‐entropy loss.Trials that use higher values of γ (for example, around 3.0 to 5.0) exhibit a much stronger focusing effect. In these cases, the loss function down-weights the loss contribution from well-classified examples, thereby emphasizing the hard examples in the minority class.
    #When γ is set to 0.0, the focal loss effectively reduces to the standard cross-entropy loss. In several trials with γ = 0, the reported top‑5 precision remains near the trivial baseline (around 0.0959).
    focal_loss_gamma = trial.suggest_float('focal_loss_gamma', 3.0, 5.0, step=0.5)  # Focuses on hard examples
    
    # It is important to note that the optimal setting for this penalty is interdependent with focal_loss_alpha and focal_loss_gamma. The best outcomes were found when relatively high α (≈0.8–1.0) and a suitably high gamma (≈3.0–4.0) were combined with a penalty in the intermediate range—this combination allows the network to focus correctly on hard examples without overcompensating.
    focal_loss_misclassified_penalty = trial.suggest_float('focal_loss_misclassified_penalty', 1.0, 10.0, step=1.0)  # Custom extension

    # Optional: Boolean switches for ablation-style trials
    #use_focal_loss = trial.suggest_categorical('use_focal_loss', [True, False])
    #undersampling_enabled = trial.suggest_categorical('undersampling', [True, False])
    #disable_test_loader = trial.suggest_categorical('disable_test_loader', [True, False])
    #model_type = trial.suggest_categorical('model_type', ['GNN', 'MLP', 'Transformer'])

    # Optional: TRECHHOLD tuning (classification threshold)
    #threshold = trial.suggest_float('threshold', 0.3, 0.9, step=0.05)
    #k_threshold = trial.suggest_int('k_threshold', 1, 10)


    # Print current trial info
    #print(f"[{datetime.now()}] Trial {trial.number}: lr={lr}, batch_size={batch_size}, hidden_dims={hidden_dims}")
    print(f"[{datetime.now()}] Trial {trial.number}: "
            f"pos_weight_factor={pos_weight_factor}, "
            f"focal_loss_alpha={focal_loss_alpha}, "
            f"focal_loss_gamma={focal_loss_gamma}, "
            f"focal_loss_misclassified_penalty={focal_loss_misclassified_penalty}")
  
    #TODO make sure val is ever empty, otherwise we will tune on specific dazasezs only 
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

  
    # Setup logging paths
    run_id = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
 
    output_dir = os.path.join(base_path, "runs", run_id) 
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "log.txt")

    # Train model and return accuracy
    acc = train_model(model, train_loader, val_loader,
                      output_neural_network=output_dir,
                      log_file_path=log_file_path, all_embeddings=COMBINATION,
                      epochs=EPOCHS, lr=lr, pos_weight_factor=pos_weight_factor,focal_loss_alpha=focal_loss_alpha, focal_loss_gamma=focal_loss_gamma , focal_loss_missclaf=focal_loss_misclassified_penalty)

    return acc

# Run Optuna
if __name__ == "__main__":

   #TODO 
   # Add a median pruner: study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner()).
    train_dataset, test_dataset, val_dataset, input_dim = create_datasets(
    random_seed=42,
    sample_subset=SAMPLE_SUBSET,
    input_path=input_path, 
    directory_path_data= directory_path_data)

    networktype ="attention"

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, input_dim, networktype), n_trials=TRIALS)  #CHANGED


    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)
