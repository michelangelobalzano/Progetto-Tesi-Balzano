import torch
import torch.nn as nn

# Criterio di minimizzazione dell'errore
criterion = nn.MSELoss()

# Calcolo dell'RMSE
def masked_prediction_loss(outputs, targets, masks):
    loss = 0
    for output, target, mask in zip(outputs, targets, masks):
        loss += torch.sqrt(criterion(output * mask, target * mask))
    return loss / len(outputs)