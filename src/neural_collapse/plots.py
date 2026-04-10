import matplotlib.pyplot as plt
from typing import List, Union

def make_plots(
    values: List[Union[float, List[float]]],
    layers: List[str],
    header: str,
    values_name: str,
    save_path: str,
    classes_names: List[str] | None = None,
    selected_layers: List[str] | None = None,
):
    plt.figure(figsize=(16, 8))

    if all(isinstance(v, (int, float)) for v in values):
        x = range(len(values))
        label = classes_names[0] if classes_names else values_name
        plt.plot(x, values, label=label, marker='o')

    else:
        # Несколько серий
        for i, series in enumerate(values):
            x = range(len(series))
            label = (
                classes_names[i]
                if classes_names and i < len(classes_names)
                else f"{i}"
            )
            plt.plot(x, series, label=label, marker='o')

    plt.grid(True)

    if selected_layers:
        selected_indices = [layers.index(l) for l in selected_layers if l in layers]
        plt.xticks(ticks=selected_indices, labels=[layers[i] for i in selected_indices], rotation=45)
    else:
        plt.xticks(range(len(layers)), layers, rotation=45)
    plt.xlabel("Layers")
    plt.ylabel(values_name)

    plt.title(header)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
