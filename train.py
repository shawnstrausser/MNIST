import torch

from config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_NAME, EXPERIMENTS_DIR
from models.registry import get_model
from training.trainer import train_one_epoch
from evaluation.evaluate import evaluate
from utils.data import get_data_loaders


def main():
    print(f"Model: {MODEL_NAME} | Device: {DEVICE} | Epochs: {EPOCHS}")
    print("-" * 50)

    train_loader, test_loader = get_data_loaders()
    model = get_model(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, DEVICE)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    # Save model
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    save_path = EXPERIMENTS_DIR / f"{MODEL_NAME}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
