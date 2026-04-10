from neural_collapse.utils import check_parameters
import matplotlib.pyplot as plt


res = check_parameters()

plt.figure(figsize=(16, 8))


for label, series in res.items():
    x = range(len(series))
    plt.plot(x, series, label=label, marker='o', linestyle='None')

plt.grid(True)

plt.xlabel("Layer")
plt.ylabel("Features")

plt.title("Number of features")
plt.legend()

plt.tight_layout()
plt.savefig("features.pdf")
