# Implementation of Particle Semi-Implicit Variational Inference

This repository contains the implementation for the [paper](https://arxiv.org/abs/2407.00649):

```
Particle Semi-Implicit Variational Inference
Jen Ning Lim and Adam M. Johansen
NeurIPS 2024
```

## Reproducing Experiments

This repository contains scripts that can be used to reproduce the experiments conducted in our research project. Follow the instructions below to reproduce the experiments on your local machine.

We provide information about the environment in `Pipfile` and `Pipfile.lock`. One can recreate this with by running `pipenv sync`.

## Running the Experiments

To reproduce the experiments, follow these steps:

* (Section 5.1) Execute the command `./scripts/sec_5.1/run.sh`.
* (Section 5.2) Execute the command `./scripts/sec_5.2/run.sh`.
* (Section 5.3) Execute the command `./scripts/sec_5.3/run.sh`.
* (Section 5.4) Execute the command `./scripts/sec_5.4/run.sh` requires `wandb` to visualize. 
