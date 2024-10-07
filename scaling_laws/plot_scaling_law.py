import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from scaling_laws.data_fns import get_scaling_law_data

data_list, fn = get_scaling_law_data()
x = np.linspace(1, 60, num=50)
y = fn(x)

# https://colorbrewer2.org/#type=diverging&scheme=RdBu&n=6
plot_map = {
    0.25: {"color": "#2166ac", "label": "B1", "params": "250K"},
    0.5: {"color": "#2166ac", "label": None, "params": "500K"},
    1.: {"color": "#2166ac", "label": None, "params": "1M"},
    2.: {"color": "#67a9cf", "label": "B2", "params": "2M"},
    4.: {"color": "#fddbc7", "label": "B3", "params": "4M"},
    16.: {"color": "#fddbc7", "label": None, "params": "16M"},
    64.: {"color": "#fddbc7", "label": None, "params": "64M"},
    256.: {"color": "#ef8a62", "label": "B4", "params": "256M"},
    1024.: {"color": "#ef8a62", "label": None, "params": "1024M"},
    32_768.: {"color": "#b2182b", "label": "B5", "params": "32B"},
}

sns.set(style="whitegrid", font_scale=2.0)
plt.figure(dpi=100, figsize=(10, 8))
for data in data_list:
    params = data["params"]
    color = plot_map[params]["color"]
    label = plot_map[params]["label"]
    plt.plot(data["compute"], data["loss"], lw=2, alpha=0.5, c=color)
    plt.scatter(data["compute"], data["loss"], lw=3, c=color, label=label, marker="o")
plt.plot(x, y, color="black", linestyle="--", lw=3)
plt.text(2, 0.002, r"$3 e^{-0.15 C}$", fontsize=25, color="black")
plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Compute (PFLOPs)")
plt.legend()
plt.tight_layout()
plt.show()

# Top of plot legend.
cases = [val["params"] for param, val in plot_map.items() if val["label"] is not None]
fig, ax = plt.subplots(dpi=100, figsize=(len(cases) * 2.5, 1))
legend_lines = []
for param, val in plot_map.items():
    if val["label"] is not None:
        color = val["color"]
        ax.plot([], [], color=color)
        legend_lines.append(Line2D([0], [0], color=color, lw=5))
ax.legend(legend_lines, cases, loc="center", ncol=len(cases), fontsize=20)
ax.set_axis_off()
plt.tight_layout()
plt.show()
