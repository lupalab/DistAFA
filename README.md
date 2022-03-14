# Distribution Guided Active Feature Acquisition

This is the official code repo for `Distribution Guided Active Feature A cquisition`.

## Code Structure

- datasets: define the datasets and dataloader
- envs: environment for RL based acquisition
  - img: image data acquisition with a fixed budget
  - img_term: image data acquisition with a given cost for each feature
  - vec: vector data acquisition with a fixed budget
  - vec_term: vector data acquisition with a given cost for each feature
- models: surrogate model and runner
  - acflow: ACFlow for 2D images
  - actan: ACFlow for 1D vectors
  - acflow_classifier: class conditioned ACFlow for 2D images
  - actan_classifier: class conditioned ACFlow for 1D vectors
  - runner: train and evaluate the model
  - model_wrapper: model wrapper for RL agent
  - group: action space grouping using a trained model
- detectors: OOD detector
  - PO3D: partially observed OOD detector
  - po3d_wrapper: detector wrapper for RL agent
- agents: AFA agents
  - img_cls_rand: random policy for image classification
  - img_cls_ppo: PPO policy for image classification
  - img_cls_hppo: hierarchical PPO policy with action space grouping
  - img_cls_hgcppo: sub-goal conditioned policy (sub-goal is a pair of classes)
  - img_cls_hgkppo: sub-goal conditioned policy (sub-goal is a set of clusters)
  - img_rec_rand: random policy for image reconstruction
  - img_rec_ppo: PPO policy for image reconstruction
  - img_rec_hppo: hierarchical PPO policy with action space grouping
  - img_rec_hgkppo: sub-goal conditioned policy (sub-goal is a set of clusters)
  - vec_cls_ppo; PPO policy for vector data classification
  - vec_cls_hppo: hierarchical PPO policy with action space grouping
  - vec_cls_gkppo: sub-goal conditioned policy (sub-goal is a set of clusters)
- scripts: running scripts
  - run_model: train the surrogate model
  - run_group: action space grouping
  - run_agent: training the AFA agent
- utils: utility functions 

## Requirements

```text
scikit-learn
tensorflow-gpu==1.14.0
tensorflow-probability==0.7.0
```

## Usage

**Please see `exp` folder for examples of the config files.**

### Train the Surrogate Model

```python
python scripts/run_model.py --cfg_file=path/to/config --mode=train
```

### Action Space Grouping

```python
python scripts/run_group.py --cfg_file=path/to/config
```

### Train OOD Detector

```text
see `detectors/PO3D/README.md`
```

### Train GMM Model

```text
use `sklearn`, see a code snippet in `exp/mnist/gmm.run.py`
```

### Train the AFA Agent

```python
python scripts/run_agent --cfg_file=path/to/config --mode=train
```
