import numpy as np
import torch.nn as nn
import torch
import time
from IPython import display
import matplotlib.pyplot as plt


def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()        # gradient initialization
        output = model(batch_X)      # forward propagation
        loss = criterion(output, batch_y) # loss calculation
        loss.backward()              # backward propagation
        optimizer.step()             # weight updates
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader) 


def evaluate(model, criterion, test_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad(): 
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            output = model(batch_X)
            loss = criterion(output, batch_y)
            
            total_loss += loss.item()
            
    return total_loss / len(test_loader)


def train_cycle(model, optimizer, criterion, train_loader, test_loader, n_epochs, device, scheduler=None):
    train_loss_log = []
    test_loss_log = []
    start_time = time.time()

    for epoch in range(n_epochs):
        # --- 1. Train & Evaluate ---
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        test_loss = evaluate(model, criterion, test_loader, device)
        
        if scheduler:
            scheduler.step()

        # --- 2. Chart updates ---
        train_loss_log.append(train_loss)
        test_loss_log.append(test_loss)
        
        display.clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_log, label='Train MSE', color='royalblue', lw=2)
        plt.plot(test_loss_log, label='Test MSE', color='darkorange', lw=2)
        plt.title(f"Training Progress [Epoch {epoch+1}/{n_epochs}]")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # --- 3. Progress ---
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        print(f"Running Time: {elapsed:.1f}s")

    print("\n Learning finished!")
    return train_loss_log, test_loss_log


def plot_prediction(model, test_loader, scaler, device="cpu"):
# estimated price vs real price
    model.eval()
    preds = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X)
            
            preds.append(output.cpu().numpy())
            actuals.append(batch_y.numpy())

    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    # ---  (Inverse Transform) ---

    def denormalize(data):
        dummy = np.zeros((len(data), 7))
        dummy[:, 3] = data.flatten() # index3 : close
        return scaler.inverse_transform(dummy)[:, 3]

    final_preds = denormalize(preds)
    final_actuals = denormalize(actuals)

    plt.figure(figsize=(12, 6))
    plt.plot(final_actuals, label='Actual Price', color='royalblue', alpha=0.7)
    plt.plot(final_preds, label='Predicted Price', color='crimson', linestyle='--')
    plt.title('Final Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return final_preds, final_actuals

# def train_cycle(model, optimizer, criterion, train_loader, test_loader, n_epochs, device, scheduler=None):
#     train_loss_log, test_loss_log = [], []
#     test_mae_log = []
    
#     
#     mae_criterion = torch.nn.L1Loss()
#     start_time = time.time()

#     for epoch in range(n_epochs):
#         # --- 1. Training Phase ---
#         model.train()
#         train_loss = 0
#         for batch_X, batch_y in train_loader:
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#             optimizer.zero_grad()
#             output = model(batch_X)
#             loss = criterion(output, batch_y)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
        
#         # --- 2. Evaluation Phase ---
#         model.eval()
#         test_loss = 0
#         test_mae = 0
#         with torch.no_grad():
#             for batch_X, batch_y in test_loader:
#                 batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#                 output = model(batch_X)
#                 test_loss += criterion(output, batch_y).item()
#                 test_mae += mae_criterion(output, batch_y).item()
        
#         if scheduler:
#             scheduler.step()

#         # log update
#         train_loss_log.append(train_loss / len(train_loader))
#         test_loss_log.append(test_loss / len(test_loader))
#         test_mae_log.append(test_mae / len(test_loader))

#         plot_training_results(train_loss_log, test_loss_log, test_mae_log)

#         elapsed = time.time() - start_time
#         print(f"Epoch {epoch+1}/{n_epochs} | Loss: {train_loss_log[-1]:.6f} | MAE: {test_mae_log[-1]:.6f} | Time: {elapsed:.1f}s")

#     return train_loss_log, test_loss_log, test_mae_log


#     def plot_training_results(train_loss_log, test_loss_log, test_mae_log):

#     epochs = np.arange(1, len(train_loss_log) + 1)
    
#     fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#     display.clear_output(wait=True)

#     # -------------------------------
#     # (1) Loss Plot (MSE)
#     # -------------------------------
#     ax[0].plot(epochs, train_loss_log, c='blue', label='Train Loss (MSE)')
#     ax[0].plot(epochs, test_loss_log, c='orange', label='Test Loss (MSE)')
#     ax[0].set_title('Training / Test Loss')
#     ax[0].set_xlabel('Epoch')
#     ax[0].set_ylabel('Loss')
#     ax[0].legend()
#     ax[0].grid(True)

#     # -------------------------------
#     # (2) MAE Plot
#     # -------------------------------
#     ax[1].plot(epochs, test_mae_log, c='green', label='Test MAE')
#     ax[1].set_title('Test Mean Absolute Error')
#     ax[1].set_xlabel('Epoch')
#     ax[1].set_ylabel('MAE')
#     ax[1].legend()
#     ax[1].grid(True)

#     plt.tight_layout()
#     plt.show()