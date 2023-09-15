import torch.nn as nn


class MLPModel(nn.Module):

    def __init__(self, input_size, hidden_units, n_classes, activation):

        super(MLPModel, self).__init__()

        layers = []
        n_units = [input_size] + hidden_units + [n_classes]
        for i, (in_units, out_units) in enumerate(zip(n_units[:-1], n_units[1:])):
            layers.append(
                nn.Linear(in_features=in_units, out_features=out_units)
            )
            if i < len(n_units) - 2:
                layers.append(activation())

        self._layers = nn.Sequential(*layers)

    def forward(self, X):
        bs = X.shape[0]
        return self._layers.forward(X.reshape(bs, -1))