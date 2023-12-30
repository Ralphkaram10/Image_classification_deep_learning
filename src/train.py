"""Module to train an image classifier"""
from __future__ import print_function
from dataclasses import dataclass
import torch
from torch import nn
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR
from DL.classification_dl import CustomDataset
from models.model import resnet18 as Net
from common.utils import load_yaml_config
from common import datakeywords as dk

cfg_train = load_yaml_config("src/config/config_train.yaml")

@dataclass
class HyperParametersTrainOneEpoch:
    """A data class for hyperparameters"""

    optimizer: Optimizer
    epoch: int
    log_interval: int = 10
    dry_run: bool = False


def train_one_epoch(
    model, device, train_loader, hyperparameters: HyperParametersTrainOneEpoch
):
    """Trains on train_loader for one epoch"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        hyperparameters.optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        hyperparameters.optimizer.step()
        if batch_idx % hyperparameters.log_interval == 0:
            print(
                f"Train Epoch: {hyperparameters.epoch} "
                f"[{batch_idx*len(data)}/{len(train_loader.dataset)}"
                f" ({100. * batch_idx / len(train_loader):.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
            )
            if hyperparameters.dry_run:
                break


def test(model, device, test_loader):
    """Tests model on test_loader"""
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: "
        f"Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({100.0 * correct / len(test_loader.dataset):.0f}%)\n"
    )


def get_torch_device():
    """Gets torch device"""
    use_cuda = cfg_train[dk.USE_CUDA_KEY] and torch.cuda.is_available()
    use_mps = cfg_train[dk.USE_MPS_KEY] and torch.backends.mps.is_available()
    torch.manual_seed(cfg_train[dk.SEED_KEY])
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def get_dataloader_kwargs(batch_size=10, num_workers=1, pin_memory=True, shuffle=True):
    """Gets dataloader keyword arguments"""
    use_cuda = cfg_train[dk.USE_CUDA_KEY] and torch.cuda.is_available()
    kwargs = {"batch_size": batch_size}
    if use_cuda:
        cuda_kwargs = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "shuffle": shuffle,
        }
        kwargs.update(cuda_kwargs)
    return kwargs


def main():
    """Main function"""

    dataset1 = CustomDataset(cfg_train[dk.TRAIN_MANIFEST_PATH_KEY])
    dataset2 = CustomDataset(cfg_train[dk.TEST_MANIFEST_PATH_KEY])

    train_loader_kwargs = get_dataloader_kwargs(
        batch_size=cfg_train[dk.BATCH_SIZE_KEY],
        num_workers=1,
        pin_memory=True,
        shuffle=True,
    )
    test_loader_kwargs = get_dataloader_kwargs(
        batch_size=cfg_train[dk.TEST_BATCH_SIZE_KEY],
        num_workers=1,
        pin_memory=True,
        shuffle=True,
    )

    train_loader = torch.utils.data.DataLoader(dataset1, **train_loader_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_loader_kwargs)

    device = get_torch_device()

    model = Net(num_classes=cfg_train[dk.NUM_CLASSES_KEY]).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=cfg_train[dk.LR_KEY])

    scheduler = StepLR(optimizer, step_size=1, gamma=cfg_train[dk.GAMMA_KEY])
    for epoch in range(1, cfg_train[dk.EPOCHS_KEY] + 1):
        hyperparameters = HyperParametersTrainOneEpoch(
            optimizer, epoch, cfg_train[dk.LOG_INTERVAL_KEY], cfg_train[dk.DRY_RUN_KEY]
        )
        train_one_epoch(model, device, train_loader, hyperparameters)
        test(model, device, test_loader)
        scheduler.step()

    if cfg_train[dk.SAVE_MODEL_KEY]:
        torch.save(model.state_dict(), "output/mnist_resnet.pt")


if __name__ == "__main__":
    main()
