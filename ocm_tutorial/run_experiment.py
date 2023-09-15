import matplotlib.pyplot as plt

import ocm_tutorial.data as data
from ocm_tutorial.utils import setup_logging, make_temp_dir
from ocm_tutorial.utils.metric import compute_metric, get_best_predictions, get_worst_predictions
from ocm_tutorial.utils.architecture import save_model_architecture
from ocm_tutorial.model.mlp_model import MLPModel
from ocm_tutorial.model.cnn_model import CNNModel
import logging
from torch.utils.data import DataLoader, random_split
import torch
import sklearn.metrics
import os


def plot_prediction(X, y_true, y_pred, filename=None):

    plt.imshow(X, cmap="Greys")
    plt.title(f"Predicted: {y_pred} Ground Truth:{y_true}")

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def main():

    make_temp_dir()
    setup_logging("./temp/log.txt")

    batch_size = 256
    learning_rate = 1e-4
    n_epochs = 10


    logging.info("Welcome to this year's OCM tutorial :)")

    training_dataset = data.MNIST(train=True)
    training_dataset, validation_dataset = random_split(training_dataset, [0.9, 0.1])
    test_dataset = data.MNIST(train=False)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #model = MLPModel(input_size=28*28, hidden_units=[128, 128], n_classes=10, activation=torch.nn.ReLU)
    model = CNNModel(input_size=28,conv_filters=[32,64], kernel_sizes=[3,3], pooling_size=2, hidden_units=[128], n_classes=10)
    save_model_architecture(model,filename="./temp/architecture.txt")

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cross_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):

        epoch_loss = 0.0
        for X,y in train_loader:

            loss = cross_entropy(model(X), y)

            loss.backward()
            opt.step()

            epoch_loss += loss

        logging.info(f"Epoch {epoch}: avg. loss={epoch_loss / len(train_loader) : .2f} nats")

        val_acc = compute_metric(val_loader, model, metric=sklearn.metrics.accuracy_score)
        logging.info(f"Epoch {epoch}: validation acc={ val_acc * 100 : .2f}%")

    test_acc = compute_metric(test_loader, model, metric=sklearn.metrics.accuracy_score)
    logging.info("Training done.")
    logging.info(f"test acc={ test_acc * 100 : .2f}%")

    best_test_pred_probs, best_test_pred_idx = get_best_predictions(test_loader, model)
    logging.info(f"Best predictions: {best_test_pred_probs}")

    worst_test_pred_probs, worst_test_pred_idx = get_worst_predictions(test_loader, model)
    logging.info(f"Worst predictions: {worst_test_pred_probs}")

    os.mkdir("./temp/predictions")

    # save the best test predictions
    for i, (prob, idx) in enumerate(zip(best_test_pred_probs.tolist(), best_test_pred_idx.tolist())):
        X, y = test_dataset[idx]
        plot_prediction(X.squeeze(),
                        y_true=y,
                        y_pred=model(X).argmax(dim=-1).item(),
                        filename=f"./temp/predictions/best_{i}.pdf")

    # save the worst test predictions
    for i, (prob, idx) in enumerate(zip(worst_test_pred_probs.tolist(), worst_test_pred_idx.tolist())):
        X, y = test_dataset[idx]
        plot_prediction(X.squeeze(),
                        y_true=y,
                        y_pred=model(X).argmax(dim=-1).item(),
                        filename=f"./temp/predictions/worst_{i}.pdf")


if __name__ == "__main__":
    main()


