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
    layer_fetures_change: list[bool] | None = None,
):
    fig, (ax, ax2) = plt.subplots(
        2, 1,
        figsize=(16, 10),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True
    )

    # =========================
    # --- MAIN PLOT (values) ---
    # =========================

    valid_x = []
    valid_y = []

    for i, v in enumerate(values):
        if v is None:
            continue
        if isinstance(v, (float, int)):
            valid_x.append(i)
            valid_y.append(v)
        else:
            # если список — берём первое значение (допущение)
            valid_x.append(i)
            valid_y.append(v[0] if len(v) > 0 else 0)

    if valid_x:
        ax.plot(valid_x, valid_y, linewidth=1, color="black", alpha=0.6)

    # scatter по типам
    unique_types = list(dict.fromkeys(layers_types))

    for t in unique_types:
        xs, ys = [], []

        for i, lt in enumerate(layers_types):
            if lt != t:
                continue

            v = values[i]
            if v is None:
                continue

            if isinstance(v, (float, int)):
                xs.append(i)
                ys.append(v)
            else:
                xs.append(i)
                ys.append(v[0] if len(v) > 0 else 0)

        if xs:
            ax.scatter(xs, ys, s=60, label=t)

    # None
    none_x = [i for i, v in enumerate(values) if v is None]
    if none_x:
        ax.scatter(
            none_x,
            [0] * len(none_x),
            marker='x',
            s=80,
            color='red',
            label='None'
        )

    ax.legend(title="Layer type")
    ax.grid(True, color="#A7C7E7")
    ax.set_ylabel(values_name)
    ax.set_title(header)

    # =========================
    # --- FEATURE CHANGE BAR ---
    # =========================

    if layer_fetures_change is not None:
        up_x, down_x = [], []

        for i, v in enumerate(layer_fetures_change):
            if v is True:
                up_x.append(i)
            elif v is False:
                down_x.append(i)

        if up_x:
            ax2.scatter(up_x, [1] * len(up_x), marker=r'$\checkmark$', color='green', s=120)

        if down_x:
            ax2.scatter(down_x, [1] * len(down_x), marker='x', color='red', s=80)

        ax2.set_ylim(0.5, 1.5)
        ax2.set_yticks([])
        ax2.set_ylabel("out_feat >= input_feat")

    else:
        ax2.axis("off")

    # =========================
    # --- X axis formatting ---
    # =========================

    n = len(layers_types)
    step = 5

    xticks = list(range(0, n, step))
    xticks_lab = list(range(start_layer, start_layer + n, step))

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks_lab, rotation=45)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
