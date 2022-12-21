import numpy as np
import torch
import matplotlib.pyplot as plt
from unbalancedot.entropy import Balanced, KullbackLeibler, TotalVariation, \
    Range, PowerEntropy

x = torch.linspace(-3, 3, 200)
x_, y_ = np.zeros(200), np.zeros(200)
setting = 2


if setting == 1:
    L_entropy = [Balanced(1e0), KullbackLeibler(1e0, 1e0),
                TotalVariation(1e0, 1e0), Range(1e0, 0.5, 2)]
    L_name = ['Balanced', 'KL', '$RG_{[0.5,2]}$', 'TV']
    L_color = ['red', 'blue', 'purple', 'green']
    L_linestyle = ['solid', 'dotted', 'dashed', 'dashdot']
if setting == 2:
    L_entropy = [KullbackLeibler(1e0, 1e0),
                PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)]
    L_name = ['KL', 'Berg', 'Hellinger']
    L_color = ['blue', 'green', 'purple']
    L_linestyle = ['dotted', 'dashed', 'dashdot']

fig = plt.figure(figsize=(4,4))
for entropy, name, color, lstyle in zip(L_entropy, L_name, L_color, L_linestyle):
    aprox = entropy.aprox
    x_, y_ = x.data.numpy(), (- aprox(-x)).squeeze().data.numpy()
    plt.plot(x_, y_, label=name, c=color, linestyle=lstyle)

plt.xlabel('p', fontsize=16)
plt.ylabel('-aprox(-p)', fontsize=16)
plt.legend(fontsize=13)
plt.grid()
plt.tight_layout()
plt.savefig(f'output/fig_aprox_{setting}.pdf', format='pdf', transparent=True)
plt.show()
