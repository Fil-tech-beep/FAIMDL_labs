import torch
from torch import nn
import wandb

from dataset.dataset_and_dataloader import build_dataloaders
from models.personalised_AlexNet import AlexNet
from eval import evaluate




def train(num_epochs, model, train_loader, val_loader, device, optimizer, loss_fn, save_path):
    best_val_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()

        running_loss = 0.0
        total_correct_predictions = 0
        total_predictions = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_predicted = model(x)
            loss = loss_fn(y_predicted, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predictions = y_predicted.argmax(dim=1)
            total_correct_predictions += (predictions == y).sum().item()
            total_predictions += y.shape[0]

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * total_correct_predictions / total_predictions



        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)




        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Saved best validation model -> {save_path}")

    return best_val_accuracy

















if __name__ == "__main__":
    data_root = "data"
    num_workers = 0
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rates = [0.1, 0.001, 0.0001]
    batch_sizes = [16, 32, 64]
    weight_decays = [0.0, 1e-4, 1e-3]


    for lr in learning_rates:
        for batch_size in batch_sizes:
            for weight_decay in weight_decays:
                run_name = f"alexnet_lr{lr}_bs{batch_size}_wd{weight_decay}"

                train_loader, val_loader = build_dataloaders(
                    data_root=data_root,
                    batch_size=batch_size,
                    num_workers=num_workers
                )

                model = AlexNet().to(device)
                loss_fn = nn.CrossEntropyLoss()

                optimizer = torch.optim.SGD(
                    params=model.parameters(),
                    lr=lr,
                    momentum=0.9,
                    weight_decay=weight_decay
                )

                save_path = f"checkpoints/{run_name}.pth"

                wandb.init(
                    project="lab-4-AlexNet",
                    name=run_name,
                    config={
                        "epochs": num_epochs,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "weight_decay": weight_decay,
                        "optimizer": "SGD",
                        "momentum": 0.9,
                        "architecture": "AlexNet",
                        "dataset": "TinyImageNet"
                    }
                )

                best_val_accuracy = train(
                    num_epochs=num_epochs,
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    save_path=save_path
                )

                print(
                    f"Finished run: {run_name} | Best Val Acc: {best_val_accuracy:.2f}%"
                )

                wandb.finish()