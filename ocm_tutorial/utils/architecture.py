import torch


def save_model_architecture(model: torch.nn.Module, filename:str="./architecture.txt"):

    with open(filename, "w") as f:
        f.write(repr(model))


