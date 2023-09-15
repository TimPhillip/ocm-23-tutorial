import torch.nn
import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self, input_size, conv_filters, kernel_sizes, pooling_size, hidden_units, n_classes):

        super(CNNModel, self).__init__()

        conv_layers = []
        channels = [1] + conv_filters
        for i, (in_channels, out_channels, kernel) in enumerate(zip(channels[:-1], channels[1:], kernel_sizes)):
            conv_layers.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel,
                          stride=1,
                          padding=0
                          )
            )
            conv_layers.append(
                torch.nn.ReLU()
            )

        conv_layers.append(nn.MaxPool2d(kernel_size=pooling_size))

        self._size_after_conv_layers = int(((input_size - torch.sum(torch.as_tensor(kernel_sizes) - 1))**2 * conv_filters[-1] / pooling_size**2).item())

        layers = []
        n_units = [self._size_after_conv_layers] + hidden_units + [n_classes]
        for i, (in_units, out_units) in enumerate(zip(n_units[:-1], n_units[1:])):
            layers.append(
                nn.Linear(in_features=in_units, out_features=out_units)
            )
            if i < len(n_units) - 2:
                layers.append(torch.nn.ReLU())

        self._conv_layers = nn.Sequential(*conv_layers)
        self._fc_layers = nn.Sequential(*layers)

    def forward(self, X):
        filter_activations = self._conv_layers.forward(X)
        logits = self._fc_layers.forward(torch.flatten(filter_activations, -3))
        return logits