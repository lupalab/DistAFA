# PO^3D: Partially Observed Out-of-Distribution Detection

code is adapted from `https://github.com/ermongroup/ncsnv2`, please cite their work if you use this repo.

## Run:

- step 1: train a ncsn network using partially observed instances

```
python ncsn_main.py --config=path/to/config --doc=path/to/log
```

- step 2: train a generatiev model over the score norms

```
python ood_main.py --config=path/to/config --doc=path/to/log
```