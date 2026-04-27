from neural_collapse.metric import compute_layers_metrics
from neural_collapse.models import ModelWrapper
from neural_collapse.trainer import Trainer
from neural_collapse.plots import make_plots
from neural_collapse.utils import get_layers
from neural_collapse.dataloader import get_dataloaders


def launch_neuro_collapse()
    dataloader_train, dataloader_test, NUM_CLASSES, ONE_CHANNEL = get_dataloaders("cifar10", batch_size=16)
    model = ModelWrapper("vgg11", NUM_CLASSES, ONE_CHANNEL, None)


    trainer = Trainer(model, dataloader_train, None, dataloader_test)
    #trainer.train_function(num_epochs=1, save_model_weigths="artifacts/cifar10_vgg11.pth", verbose=True)
    #trainer.eval_function(verbose=True)

    layers_names, layers_types = get_layers(trainer.model.model)

    values = compute_layers_metrics(layers_names, model, NUM_CLASSES, dataloader_test, True)

    make_plots(
        values=values.tolist(),
        layers=layers_names,
        header="Resnet18 on Cifar10",
        values_name="Metric",
        save_path="metric_all.pdf",
        classes_names=None,
        selected_layers=['features.0', 'features.5', 'features.10', 'features.15', 'features.20', 'classifier.6'],
    )


if __name__ == "__main__":
    launch_neuro_collapse()
