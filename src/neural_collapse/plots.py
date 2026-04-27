import matplotlib.pyplot as plt
from typing import List, Union
import torch 


def make_plots(
    values: List[Union[float, None, List[float]]],
    layers_types: list[str],
    header: str,
    values_name: str,
    save_path: str,
    start_layer: int = 0,
):
    plt.figure(figsize=(16, 8))

    # --- line values (None исключаем) ---
    valid_x = [i for i, v in enumerate(values) if not torch.isnan(torch.tensor(v))]
    valid_y = [v for v in values if not torch.isnan(torch.tensor(v))]

    plt.plot(valid_x, valid_y, linewidth=1, color="black", alpha=0.6)

    # --- scatter по типам ---
    unique_types = list(dict.fromkeys(layers_types))

    for t in unique_types:
        idxs = [i for i, lt in enumerate(layers_types) if lt == t]

        xs = []
        ys = []

        for i in idxs:
            v = values[i]
            if v is not None:
                xs.append(i)
                ys.append(v)

        if xs:
            plt.scatter(xs, ys, s=60, label=t)

    # --- None как кресты около 0 ---
    none_x = [i for i, v in enumerate(values) if torch.isnan(torch.tensor(v))]
    if none_x:
        plt.scatter(
            none_x,
            [0 for _ in none_x],
            marker='x',
            s=80,
            color='red',
            label='None'
        )

    # --- xticks ---
    n = len(layers_types)
    step = 5
    xticks = list(range(0, n, step))
    xticks_lab = list(range(start_layer, start_layer + n, step))
    plt.xticks(xticks, xticks_lab, rotation=45)

    plt.legend(title="Layer type")

    plt.grid(True, color="#A7C7E7")

    plt.xlabel("Layer index")
    plt.ylabel(values_name)
    plt.title(header)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
