import argparse
import neural_collapse
from neural_collapse.utils import get_forward_trace


def main():
    parser = argparse.ArgumentParser(description="Experiments for neural collapse.")
    parser.add_argument("model_name", type=str, help="Model name.")
    parser.add_argument("dataset_name", type=str, help="Dataset name.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for datalaoders.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Model weights.",
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        default=0,
        help="From which layer should start.",
    )
    parser.add_argument(
        "--train", type=str, nargs="?", default=False, help="Need to train model (set with path to save weights)."
    )
    parser.add_argument(
        "--verbose",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--linear",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--affine",
        action="store_const",
        const=True,
        default=False,
    )


    args = parser.parse_args()

    ARTIFACT_FOLDER = "artifacts/"

    import torch
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    dataloader_train, dataloader_test, NUM_CLASSES, ONE_CHANNEL = neural_collapse.get_dataloaders(args.dataset_name, batch_size=args.batch_size)
    weights_path = ARTIFACT_FOLDER + args.weights if args.weights else args.weights
    model = neural_collapse.ModelWrapper(args.model_name, NUM_CLASSES, ONE_CHANNEL, weights_path)


    trainer = neural_collapse.Trainer(model, dataloader_train, None, dataloader_test)
    if args.train:
        trainer.train_function(num_epochs=5, save_model_weigths=ARTIFACT_FOLDER + args.train, verbose=args.verbose)
    trainer.eval_function(verbose=args.verbose)

    layers_names, layers_types, layer_fetures_change = get_forward_trace(trainer.model.model, args.start_layer)

    print(layers_names, layers_types)

    values = neural_collapse.compute_layers_metrics(layers_names, trainer.model.model, NUM_CLASSES, dataloader_test, args.verbose, args.linear, args.affine)

    neural_collapse.make_plots(
        values=values.flatten().tolist(),
        layers_types=layers_types,
        header=f"{args.model_name} on {args.dataset_name}",
        values_name="Metric",
        save_path=f"plots/metric_{args.model_name}_{args.dataset_name}.pdf",
        start_layer=args.start_layer,
        layer_fetures_change=layer_fetures_change,
    )


if __name__ == "__main__":
    main()
