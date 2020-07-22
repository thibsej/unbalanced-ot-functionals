import numpy as np
import torch
import matplotlib.pyplot as plt
from common.entropy import Balanced, KullbackLeibler, TotalVariation, Range, PowerEntropy

x = torch.linspace(-5, 5, 200)
L_entropy = [Balanced(1e0), KullbackLeibler(1e0, 1e0), TotalVariation(1e0, 1e0),Range(1e0, 0.5, 2),
             PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)]
L_name = ['Balanced', 'KL', '$RG_{[0.5,2]}$', 'TV', 'Berg', 'Hellinger']
for entropy, name in zip(L_entropy, L_name):
    aprox = entropy.aprox
    x_, y_ = x.data.numpy(), (- aprox( -x )).squeeze().data.numpy()
    plt.plot(x_, y_, label=name)

plt.xlabel('p', fontsize=16)
plt.ylabel('-aprox(-p)', fontsize=16)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig('output/fig_aprox.eps', format='eps', transparent=True)
plt.show()