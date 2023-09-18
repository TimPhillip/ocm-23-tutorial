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

import mlflow
import hydra
from omegaconf import OmegaConf


def plot_prediction(X, y_true, y_pred, filename=None):

    plt.imshow(X, cmap="Greys")
    plt.title(f"Predicted: {y_pred} Ground Truth:{y_true}")

    if filename is not None:
        plt.savefig(filename)

    plt.show()


@hydra.main(config_path="../config", config_name="ocm_experiment", version_base=None)
def main(cfg):

    print(OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg.logging.tracking_server)
    mlflow.set_experiment(experiment_name=cfg.logging.experiment_name)

    make_temp_dir()
    setup_logging("./temp/log.txt")

    logging.info("Welcome to this year's OCM tutorial :)")

    training_dataset = hydra.utils.instantiate(cfg.data.training)
    training_dataset, validation_dataset = random_split(training_dataset, [1.0 - cfg.data.validation_fraction,
                                                                           cfg.data.validation_fraction])
    test_dataset = hydra.utils.instantiate(cfg.data.training)

    train_loader = DataLoader(training_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = hydra.utils.instantiate(cfg.model)
    save_model_architecture(model,filename="./temp/architecture.txt")


    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    cross_entropy = torch.nn.CrossEntropyLoss()

    with mlflow.start_run() as run:

        # log the model parameters
        mlflow.log_param("learning_rate", cfg.learning_rate)
        mlflow.log_param("model_class", model.__class__.__name__)

        # tag the run
        mlflow.set_tags(tags={
            "ocm-23": "true",
            "mlflow": "used",
            "hydra": "used"
        })

        mlflow.log_artifact("./temp/architecture.txt")

        for epoch in range(cfg.epochs):

            epoch_loss = 0.0
            for X,y in train_loader:

                loss = cross_entropy(model(X), y)
                loss.backward()
                opt.step()

                epoch_loss += loss

                # log the loss value every step
                mlflow.log_metric("training/loss", value=loss.item())

            logging.info(f"Epoch {epoch}: avg. loss={epoch_loss / len(train_loader) : .2f} nats")
            mlflow.log_metric("training/epoch_loss", value=epoch_loss / len(train_loader))

            val_acc = compute_metric(val_loader, model, metric=sklearn.metrics.accuracy_score)
            logging.info(f"Epoch {epoch}: validation acc={ val_acc * 100 : .2f}%")
            mlflow.log_metric("validation/accuracy", value=val_acc * 100)

            mlflow.log_artifact("./temp/log.txt")

        # save the weights of the model
        torch.save(model, "./temp/model.pt")
        mlflow.log_artifact("./temp/model.pt")


        test_acc = compute_metric(test_loader, model, metric=sklearn.metrics.accuracy_score)
        logging.info("Training done.")
        logging.info(f"test acc={ test_acc * 100 : .2f}%")
        mlflow.log_metric("test/accuracy", value=test_acc * 100)

        best_test_pred_probs, best_test_pred_idx = get_best_predictions(test_loader, model)
        logging.info(f"Best predictions: {best_test_pred_probs}")
        mlflow.log_metric("best_predition_prob", value=best_test_pred_probs[0].item())

        worst_test_pred_probs, worst_test_pred_idx = get_worst_predictions(test_loader, model)
        logging.info(f"Worst predictions: {worst_test_pred_probs}")
        mlflow.log_metric("worst_predition_prob", value=worst_test_pred_probs[0].item())

        os.mkdir("./temp/predictions")

        # save the best test predictions
        for i, (prob, idx) in enumerate(zip(best_test_pred_probs.tolist(), best_test_pred_idx.tolist())):
            X, y = test_dataset[idx]
            plot_prediction(X.squeeze(),
                            y_true=y,
                            y_pred=model(X).argmax(dim=-1).item(),
                            filename=f"./temp/predictions/best_{i}.pdf")
            mlflow.log_artifact(f"./temp/predictions/best_{i}.pdf", "predictions")

        # save the worst test predictions
        for i, (prob, idx) in enumerate(zip(worst_test_pred_probs.tolist(), worst_test_pred_idx.tolist())):
            X, y = test_dataset[idx]
            plot_prediction(X.squeeze(),
                            y_true=y,
                            y_pred=model(X).argmax(dim=-1).item(),
                            filename=f"./temp/predictions/worst_{i}.pdf")
            mlflow.log_artifact(f"./temp/predictions/worst_{i}.pdf", "predictions")


if __name__ == "__main__":
    main()


