import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)




######## SETTINGS data creation ########
TRAIN_TEST_VAL_RATIO = "ONE_ONLY" #"PERCENTAGES" #"ONE_ONLY"  # Options: "PERCENTAGES", "ONE_ONLY",

# Only used if TRAIN_TEST_VAL_RATIO == "PERCENTAGES"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

WHAT_TO_CONSIDER = "isPredessor" #"hasChangedNeighbor"



######## SETTINGS data filtering ########

TRAIN_TEST_SPLIT ="TRAINVALTEST" # alternative "TRAINTEST" if no val set required 

REMOVE_DUPLICATES= False #widersprüche werden entfernt

ADJUST_TRAIN_SIZE= "MAKEFOLDERSEQUAL" # such that all training data has the same size 
# alternative "CUT_ABOVE", not oversampling too large stuff DO_NOT_CUT

ADJUST_TRAIN_TEST_VAL_RATIO= False #we will oversample train such that for indivudal data point the distribtuion 80, 10, 10 still holds 

#only if make folders equal 
MAKE_FOLDERS_EQUAL_SIZE_OR_CUT_ABOVE = 100000

BALANCE_METHOD = "weighted_loss"  # Options: "undersampling" or "weighted_loss" #if weirghtedloss always BCEWithLogitsLoss, 


######## SETTINGS network ########
 #  original BCE loss for undersampling
LOSSFUNCTION =  "focalLoss" #"BCEWithLogitsLoss" #:"focalLoss" # "BCEWithLogitsLoss" #"BCELoss"# "focalLoss"

# what is given to the LLM "VECTOR_ONLY" #COMBINATION == "VECTOR_AND_IDS" (COMBINATION == "ALL_EMBEDDINGS")
COMBINATION =  "VECTOR_ONLY"

