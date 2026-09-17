import torch
import wandb



def evaluate(model, val_loader, loss_fn, device):
    model.eval()

    val_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            y_predicted = model(x)
            loss = loss_fn(y_predicted, y)

            val_loss += loss.item()
            predictions = y_predicted.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.shape[0]

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100.0 * correct / total

    return val_loss, val_accuracy