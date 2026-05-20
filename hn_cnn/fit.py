##################################### 📦 Imports ##########################################

import torch  # PyTorch library for deep learning
from hn_cnn.cnn import HNANN, HNCNN, HNLR # Import model classes for Logistic Regression (HNLR), ANN (HNANN), and CNN (HNCNN)
from hn_cnn.constants import * # Import global constants (like keys: TRAIN, ROC, AUC, etc.)
from hn_cnn.utils import update_parameters, save_model # Utility functions: updating hyperparameters and saving models


##################################### ⚙️ Default Hyperparameters ##########################################

DEFAULT_HYPERPARAMETERS = {
    EPOCHS: 3000,                   # Total number of training epochs
    LEARNING_RATE: 0.05,           # Initial learning rate
    MOMENTUM: 0.9,                 # Momentum for SGD optimizer
    DAMPENING: 0,                  # Dampening term for momentum
    RELU_SLOPE: 0.01,              # Not used here, possibly for LeakyReLU
    WEIGHTS_DECAY: 0.0001,         # L2 regularization strength
    OPTIMIZER: torch.optim.SGD,    # Optimizer to use (can override with Adam, etc.)
    CLASS_WEIGHTS: [0.7, 3.7],     # Class weights for imbalance (e.g., for BCE loss)
}

# Default threshold for early stopping based on training AUC
DEFAULT_STOP_THRESHOLD = 0.95


##################################### 📈 Evaluation Function (No Gradient Calculation) ##########################################

@torch.no_grad()  # Disable gradient calculation during evaluation for speed and memory
def evaluate(model, val_loader, weights, predict=False):
    """ Evaluate the model on validation data.
    
    Args:
        model: The model to evaluate.
        val_loader: DataLoader for validation data.
        weights: Class weights for loss calculation.
        predict: If True, return predictions instead of metrics.
    
    Returns:
        metrics: Dictionary containing evaluation metrics or predictions.
    """

    model.eval()  # Set model to evaluation mode (e.g., disables dropout, BN updates)
    metrics = {}  # Dictionary to hold performance metrics

    # Iterate through the validation/test DataLoader
    for batch in val_loader:
        # Run validation step defined in the model (returns metrics like AUC, loss)
        metrics = model.validation_step(batch, weights, predict)
    return metrics  # Return computed metrics


##################################### 🔁 Training Function ##########################################

def fit(model, data_loaders, parameters={}, store_model={}):
    """ Train the model.
    """
    output = []           # To store metrics per epoch
    best_val_auc = 0      # Track best validation AUC for model saving

    # Merge user-provided parameters with defaults
    hyperparameters = update_parameters(parameters, DEFAULT_HYPERPARAMETERS)

    # If the model is Logistic Regression (not a neural network)
    if isinstance(model, HNLR):
        # Train the logistic regression model
        for batch in data_loaders[TRAIN]:
            model.training(batch, class_weights=hyperparameters[CLASS_WEIGHTS])
        
        # Compute metrics for validation and test sets
        metrics = {}

        # Evaluate on all other splits (validation, test, etc.)
        for subset, data_loader in data_loaders.items():
            if subset != TRAIN:
                for batch in data_loader:
                    subset_metrics = model.validation_step(batch)
                    metrics[subset] = subset_metrics
                    print(subset)
                    print(subset_metrics)
        output.append(metrics)  # Append metrics to output list


##################################### 🧠 Neural Network Training (CNN/ANN) ##########################################

    else:
        # Initialize optimizer with model parameters and hyperparameters
        optimizer = hyperparameters[OPTIMIZER](
            model.parameters(),
            lr=hyperparameters[LEARNING_RATE],
            momentum=hyperparameters[MOMENTUM],
            dampening=hyperparameters[DAMPENING],
            weight_decay=hyperparameters[WEIGHTS_DECAY],
        )
        # ADAM
        # optimizer = opt_func(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        # Training
        # start_time = time.time()

        # Loop over training epochs
        for epoch in range(0, hyperparameters[EPOCHS]):
            print(f"Epoch {epoch}/{hyperparameters[EPOCHS]}")
            # Training
            model.train()

            # Iterate through each batch in training set
            for batch in data_loaders[TRAIN]:
                optimizer.zero_grad() # Clear previous gradients
                loss = model.training_step(batch, hyperparameters[CLASS_WEIGHTS]) # Compute loss
                
                # Gradient Normalization
                # grad_norm = 0
                # grad_params = torch.autograd.grad(outputs=loss,
                #    inputs=model.parameters(),
                #    create_graph=True
                # )
                # for grad in grad_params:
                #    grad_norm += grad.pow(2).sum()
                # grad_norm = grad_norm.sqrt()
                # loss = loss + grad_norm
                
                # Backpropagation
                loss.backward()
                # Clipping the weights
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                
                # Update model weights
                optimizer.step()
                # Clipping the weights
                #with torch.no_grad():
                #    for param in model.parameters():
                #        param.clamp_(-2, 2)
            # Time by epoch
            # print("--- %s seconds ---" % (time.time() - start_time))

            # After training, evaluate on validation and other sets
            model.eval()
            with torch.no_grad():
                metrics = {}  # Reset metrics
                for subset, data_loader in data_loaders.items():
                    if subset != TRAIN:
                        # Run evaluation step
                        subset_metrics = evaluate(model, data_loader, hyperparameters[CLASS_WEIGHTS])
                        metrics[subset] = subset_metrics
                        print(subset)
                        print(subset_metrics)
                output.append(metrics)  # Store metrics for this epoch


##################################### 💾 Model Saving Logic (Optional) ##########################################

                # Save model only if:
                # - store_model is enabled,
                # - validation AUC improves and is above threshold,
                # - AUC gap between train and validation is within MAX_DIFFERENCE
                if store_model.get(MODEL_PATH) and metrics[VALIDATION][ROC][AUC] > best_val_auc \
                    and metrics[VALIDATION][ROC][AUC] > store_model.get(THRESHOLD, 0) \
                        and abs(metrics[VALIDATION][ROC][AUC] - metrics[TRAIN_METRICS][ROC][AUC]) < \
                            store_model.get(MAX_DIFFERENCE, 1):
                    
                    # Save the model with current epoch, optimizer state, and training loss
                    save_model(
                        store_model[MODEL_PATH],
                        epoch,
                        model,
                        optimizer,
                        metrics[TRAIN_METRICS][LOSS],
                        model_id=store_model[MODEL_ID] or str(type(model))
                    )
                    best_val_auc = metrics[VALIDATION][ROC][AUC]  # Update best validation AUC
                

##################################### ⛔ Early Stopping (Optional) ##########################################

                # Get early stopping threshold from config or use default
                stop_threshold = store_model.get(STOP_THRESHOLD, DEFAULT_STOP_THRESHOLD)

                # Stop training early if training AUC exceeds threshold
                if metrics[TRAIN_METRICS][ROC][AUC] > stop_threshold:
                    print(f"Early stop: training AUC above {stop_threshold}")
                    break

    # End of training loop
    return output  # Return list of metrics per epoch


##################################### 🔮 Prediction Function ##########################################

def predict(model, data_loaders, parameters={}, threshold=None):
    """ Make predictions on test/validation set using trained model
    """

    metrics = {} # Store metrics or predicted values
    hyperparameters = update_parameters(parameters, DEFAULT_HYPERPARAMETERS) # Update hyperparameters if any provided

    # Check if model is neural network
    if isinstance(model, HNANN) or isinstance(model, HNCNN):
        # Train the neural networks
        optimizer = hyperparameters[OPTIMIZER](
            model.parameters(),
            lr=hyperparameters[LEARNING_RATE],
            momentum=hyperparameters[MOMENTUM],
            dampening=hyperparameters[DAMPENING],
            weight_decay=hyperparameters[WEIGHTS_DECAY],
        )
        
        model.eval() # Set model to eval mode
        with torch.no_grad(): # Disable gradient calculation
            # Iterate through each batch in training set
            for subset, data_loader in data_loaders.items():
                if subset != TRAIN:
                    # Evaluate model and return predicted outputs or metrics
                    subset_metrics = evaluate(model, data_loader, hyperparameters[CLASS_WEIGHTS], predict=True)
                    metrics[subset] = subset_metrics
                    print(subset)
                    print(subset_metrics)
                    
    return metrics # Return dictionary of metrics or predictions
