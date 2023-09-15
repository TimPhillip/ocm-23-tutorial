import torchvision.datasets.mnist
from torchvision import transforms


class MNIST(torchvision.datasets.mnist.MNIST):
    
    def __init__(self, train=True):
        super(MNIST, self).__init__(
            root="data/",
            download=True,
            transform= transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            ),
            train=train
        )