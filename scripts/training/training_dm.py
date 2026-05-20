################################## 📦 Importing Required Libraries ##################################

import contextlib                                                    # Provides utilities for working with context managers, used to redirect stdout
import os                                                            # Provides a way of using operating system dependent functionality
import random                                                        # Used to generate random numbers for seeding

import numpy as np                                                   # Numerical computations and array manipulations
from sklearn.model_selection import KFold, StratifiedKFold           # For data splitting strategies
import torch                                                         # PyTorch deep learning framework
from torch.utils.data.dataloader import DataLoader                   # Used to load data in batches
# from torchsummary import summary                                   # (Commented) Used to print model summary similar to Keras

# Importing custom modules from hn_cnn package
from hn_cnn.cnn import MODELS                                                                # Dictionary of model architectures (CNN, ANN, LR)
from hn_cnn.constants import *                                                               # Constants like TRAIN, TESTING, DM, etc.
from hn_cnn.data_augmentation import get_training_augmentation, TRANSFORM_TO_TENSOR          # Data transformations
from hn_cnn.fit import fit                                                                   # Function that handles training of the model
from hn_cnn.parse_data import ImageDataset                                                   # Custom dataset class to parse image + clinical data


################################## 📁 Data and Paths ##################################

# Dictionary specifying paths for each dataset split: train, validation, and test
DATA = {
    TRAIN: {
        CLINICAL_DATA_PATH: "data/pre-processed/canada.csv",
        SCANS_PATH: "data/pre-processed/canada_c",
    },
    VALIDATION: {
        CLINICAL_DATA_PATH: "data/pre-processed/canada.csv",
        SCANS_PATH: "data/pre-processed/canada_c",
    },
    TESTING: {
        CLINICAL_DATA_PATH: "data/pre-processed/maastro.csv",
        SCANS_PATH: "data/pre-processed/maastro_c",   
    }
}



################################## ⚙️ Configuration Settings ##################################

CONFIGURATIONS = {
    MODEL: ANN,                                       # Type of model to use (CNN, ANN, LR)
    DATA_SPLIT: COHORT_SPLIT,                         # Splitting strategy: COHORT_SPLIT or CROSS_VALIDATION
    FOLDS: 5,                                         # Number of folds for cross-validation (used only if CROSS_VALIDATION)
    TIME_TO_EVENT: 2 * 365,                           # Time period (in days) to consider if the event occurred (e.g., 2 years)
    EVENT: DM,                                        # Type of event to predict: DM, LRF, OS.
    HYPERPARAMETERS: {
        LEARNING_RATE: 0.05,                          # Override default learning rate
    },
    BATCH_SIZE: 64,                                   # Number of samples per training batch
    LOGS_PATH: "results/logs/log_dm-ANN.txt",             # Directory to store log files
    CLINICAL_VARIABLES: [],                           # List of clinical variables to include (empty means none)
    DATA_AUGMENTATION: {},                            # Data augmentation parameters (empty means no augmentation)
    # Optional model saving config (commented out)
    STORE_MODEL: {
         MODEL_ID: "dm",
         MODEL_PATH: 'results/models',
         THRESHOLD: 0.8,
         MAX_DIFFERENCE: 0.03,
    }
}


################################## 🔢 Random Seed Initialization ##################################

# 1) Global python seed
random.seed(8677988)                                    # Set a fixed seed for reproducibility
# 2) Random split seed
random_seed_split = random.randint(0, 7651962)          # Random seed for splitting data into folds
# 3) Random seed torch
torch.manual_seed(5745558)                               # Set seed for reproducibility in PyTorch

# Not necessary (doesn't make a difference when reproducing the results)
# device = torch.device("cpu")
# torch.backends.cudnn.deterministic = True
# torch.set_num_threads(1)


################################## 🧪 Cohort Split Strategy ##################################

if CONFIGURATIONS[DATA_SPLIT] == COHORT_SPLIT:
    # Construct the path to the log file where training output will be saved
    file_path = CONFIGURATIONS[LOGS_PATH]
    
    # Load the complete training dataset before applying any center-based splitting
    dataset_train = ImageDataset(
        DATA[TRAIN][CLINICAL_DATA_PATH],                  # CSV file containing clinical info
        DATA[TRAIN][SCANS_PATH],                          # Folder containing CT scan image patches
        TRANSFORM_TO_TENSOR,                              # Transformation applied to the image (no augmentation here)
        timeframe=CONFIGURATIONS[TIME_TO_EVENT],          # Time threshold to define positive event label (e.g., 2 years)
        event=CONFIGURATIONS[EVENT],                      # Type of event to be predicted (e.g., DM = distant metastasis)
    )

    # Open the log file in write mode to capture all console outputs during training
    with open(file_path, "w") as o:
        
        # Redirect all standard output (e.g., print statements) to the log file
        with contextlib.redirect_stdout(o):
            # Initialize an empty dictionary to store PyTorch dataloaders for each dataset
            dataloaders = {}
            # Filter IDs belonging to training centers (HGJ and CHUS)
            train_ids = [id for id in dataset_train.keys if "HGJ" in id or "CHUS" in id]
            # Filter IDs belonging to validation centers (HMR and CHUM)
            validation_ids = [id for id in dataset_train.keys if "HMR" in id or "CHUM" in id]
            # Store the same training dataset again for computing performance metrics (without augmentation)
            DATA[TRAIN_METRICS] = DATA[TRAIN]
            # Check if the selected model is a neural network (CNN or ANN)
            is_neural_network = CONFIGURATIONS[MODEL] in [CNN, ANN]

            # Iterate over each dataset group: TRAIN, TRAIN_METRICS, VALIDATION, TESTING
            for dataset in [TRAIN, TRAIN_METRICS, VALIDATION, TESTING]:
                # Get the corresponding file paths for clinical data and scan images
                paths = DATA[dataset]
                # Initialize empty list of patient IDs for this dataset
                dataset_ids = []
                # Flag to check if current dataset is the main training set
                is_training = (dataset == TRAIN)

                # For train, train_metrics, and validation datasets, use respective patient IDs
                if dataset in [TRAIN, TRAIN_METRICS, VALIDATION]:
                    dataset_ids = train_ids if is_training or dataset == TRAIN_METRICS else validation_ids

                # Parse the dataset using ImageDataset class
                dataset_parsed = ImageDataset(
                    paths[CLINICAL_DATA_PATH],  # Path to the clinical CSV file
                    paths[SCANS_PATH],          # Path to the folder containing CT scan images
                    get_training_augmentation(augment=CONFIGURATIONS[DATA_AUGMENTATION]) if is_training else TRANSFORM_TO_TENSOR,
                        # Apply data augmentation only for the main training set
                    ids_to_use=dataset_ids,     # Only include selected patient IDs (based on center)
                    timeframe=CONFIGURATIONS[TIME_TO_EVENT],  # Time threshold for defining event occurrence
                    event=CONFIGURATIONS[EVENT],               # Event type (e.g., DM, LRF, OS)
                )

                # Wrap the parsed dataset in a DataLoader for batched training
                dataloaders[dataset] = DataLoader(
                    dataset_parsed,
                    batch_size = CONFIGURATIONS[BATCH_SIZE] if is_training and is_neural_network else len(dataset_parsed.keys),
                        # Use configured batch size for neural networks; else use all samples at once
                    shuffle = is_training,      # Shuffle training data; not for validation/test
                    drop_last = is_training,    # Drop last batch if it’s incomplete (only for training)
                )

            # Initialize the model architecture as defined in CONFIGURATIONS (e.g., CNN)
            model = MODELS[CONFIGURATIONS[MODEL]]()
            print(model) # Print the model architecture to the log file

            # Begin training the model and store the training history (metrics, losses, etc.)
            history = fit(
                model,                            # Model to train
                dataloaders,                      # Dictionary of all dataloaders (train/val/test)
                parameters=CONFIGURATIONS[HYPERPARAMETERS],  # Hyperparameters such as learning rate
                store_model=CONFIGURATIONS[STORE_MODEL],  # Configuration for saving the trained model
            )


################################## 🔁 Cross-Validation Strategy ##################################

elif CONFIGURATIONS[DATA_SPLIT] == CROSS_VALIDATION:
    # Initialize stratified k-fold cross-validation to maintain class balance in each fold
    kfold = StratifiedKFold(
        n_splits=CONFIGURATIONS[FOLDS],        # Number of folds to split into (e.g., 5)
        shuffle=True,                          # Shuffle the data before splitting
        random_state=random_seed_split         # Seed for reproducibility
    )

    # Load the entire training dataset as a single pool to be split into folds
    dataset_train = ImageDataset(
        DATA[TRAIN][CLINICAL_DATA_PATH],                   # Path to the clinical CSV
        DATA[TRAIN][SCANS_PATH],                           # Path to the scan folder
        TRANSFORM_TO_TENSOR,                               # No data augmentation during splitting
        timeframe=CONFIGURATIONS[TIME_TO_EVENT],           # Time threshold for event classification
        event=CONFIGURATIONS[EVENT],                       # Type of event (e.g., DM)
    )

    # Loop through each fold generated by StratifiedKFold
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(dataset_train, np.array(dataset_train.y))):
            # Print current fold number to stdout (will be redirected to log file)
            print(f"Fold: {fold}")
            # Create a unique log file path for this fold
            file_path = CONFIGURATIONS[LOGS_PATH].replace(".txt", f"_fold{fold}.txt")
            
            # Open the log file in write mode
            with open(file_path, "w") as o:
                
                # Redirect all standard output to the log file
                with contextlib.redirect_stdout(o):
                    # Initialize an empty dictionary to hold dataloaders for each dataset split
                    dataloaders = {}
                    # Store the same training dataset again for computing performance metrics (without augmentation)
                    DATA[TRAIN_METRICS] = DATA[TRAIN]
                    # Check if the selected model is a neural network (CNN or ANN)
                    is_neural_network = CONFIGURATIONS[MODEL] in [CNN, ANN]

                    # Iterate over each dataset group: TRAIN, TRAIN_METRICS, VALIDATION, TESTING
                    for dataset in [TRAIN, TRAIN_METRICS, VALIDATION, TESTING]:
                        # Get the paths for clinical data and scans
                        paths = DATA[dataset]
                        # Initialize empty list of patient IDs for this dataset
                        dataset_ids = []
                        # Flag to check if current dataset is the main training set
                        is_training = (dataset == TRAIN)

                        # For train, train_metrics, and validation datasets, use respective patient IDs
                        if dataset in [TRAIN, TRAIN_METRICS, VALIDATION]:
                            dataset_ids = [dataset_train.keys[i] for i in train_ids] if \
                                is_training or dataset == TRAIN_METRICS else [dataset_train.keys[i] for i in validation_ids]

                        # Parse the dataset using ImageDataset class
                        dataset_parsed = ImageDataset(
                            paths[CLINICAL_DATA_PATH],  # Path to clinical CSV file
                            paths[SCANS_PATH],          # Path to scan images folder
                            get_training_augmentation(augment=CONFIGURATIONS[DATA_AUGMENTATION]) if is_training else TRANSFORM_TO_TENSOR,
                                # Apply data augmentation only for training set; else use tensor transformation
                            ids_to_use=dataset_ids,     # Use specific patient IDs based on fold split
                            timeframe=CONFIGURATIONS[TIME_TO_EVENT],  # Time threshold for event occurrence
                            event=CONFIGURATIONS[EVENT],               # Event type (e.g., DM)
                        )

                        # Wrap dataset into PyTorch DataLoader for batch processing
                        dataloaders[dataset] = DataLoader(
                            dataset_parsed,
                            batch_size = CONFIGURATIONS[BATCH_SIZE] if is_training and is_neural_network else len(dataset_parsed.images),
                                # Use batch size for NN; else use full dataset in one batch
                            shuffle = is_training,      # Shuffle only during training
                            drop_last = is_training,    # Drop last incomplete batch if training
                        )
                    
                    # Instantiate the model using the selected configuration (CNN, ANN, LR)
                    model = MODELS[CONFIGURATIONS[MODEL]]()
                    print(model)  # Print the model architecture to the log file

                    # Begin training the model and store the training history (metrics, losses, etc.)
                    history = fit(
                        model,                            # Model to train
                        dataloaders,                      # Dictionary of all dataloaders (train/val/test)
                        parameters=CONFIGURATIONS[HYPERPARAMETERS],  # Hyperparameters such as learning rate
                        store_model={**CONFIGURATIONS[STORE_MODEL], MODEL_ID: f"dm_5-fold_cv_fold{fold}"},  # Configuration for saving the trained model
                    )
