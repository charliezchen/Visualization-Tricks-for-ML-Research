import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from scaling_laws.data_fns import get_scaling_law_data
from scaling_laws.data_fns import exponential_decay

data_list = get_scaling_law_data()
x = np.linspace(5, 70, num=50)
y = exponential_decay(x, A=3., k=0.15, C=1.0)

sns.set(style="whitegrid", font_scale=2.0, rc={"lines.linewidth": 3.0})
plt.figure(dpi=100, figsize=(10, 8))
for data in data_list:
    plt.plot(data["compute"], data["loss"])
    plt.scatter(data["compute"], data["loss"])
plt.plot(x, y, color="black")
plt.yscale("log")
plt.xscale("log")
plt.ylabel("Loss")
plt.xlabel("Compute (PFLOPs)")
plt.tight_layout()
plt.show()
