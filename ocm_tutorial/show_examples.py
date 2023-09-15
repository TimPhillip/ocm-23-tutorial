from matplotlib import pyplot as plt
import logging
from torch.utils.data import DataLoader
from itertools import islice

from ocm_tutorial.utils import setup_logging
from ocm_tutorial.data import MNIST


def main():

    setup_logging()
    dataset = MNIST(train=True)
    loader = DataLoader(dataset, shuffle=True)

    n_rows, n_cols = 3,3
    fig, axs = plt.subplots(n_rows, n_cols)
    for ax,(X,y) in zip(axs.reshape(-1), islice(loader,n_rows*n_cols)):

        ax.imshow(X.squeeze(),cmap='Greys')

    plt.show()


if __name__ == "__main__":
    main()