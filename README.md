# Unbalanced Optimal Transport functionals

This repository contains the implementation of the paper [Sinkhorn Divergences for Unbalanced Optimal Transport](https://arxiv.org/pdf/1910.12958.pdf) in pytorch. If you use this work for your research, please cite the paper:

```
@article{sejourne2019sinkhorn,
  title={Sinkhorn Divergences for Unbalanced Optimal Transport},
  author={S{\'e}journ{\'e}, Thibault and Feydy, Jean and Vialard, Fran{\c{c}}ois-Xavier and Trouv{\'e}, Alain and Peyr{\'e}, Gabriel},
  journal={arXiv preprint arXiv:1910.12958},
  year={2019}
}
```
## The repository
This repository allows to compute the entropically regularized optimal transport in both balanced and unbalanced settings, with divergences such as Kullback-Leibler and total variation.

All functionals such as regularized OT, Sinkhorn divergence and maximum mean discrepancy is available in [`common/functional.py`](common/functional.py). 

See [`examples/plot_unb_gradient_flows_2D_frame.py`](examples/plot_unb_gradient_flows_2D_frame.py) for an example.
