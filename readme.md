## Code for the paper "HyFAD: Hybrid Time-Frequency Diffusion with Frequency-Aware Embedding for Time Series Imputation"


### Requirements

```bash
pip install -r requirements.txt
```

### Dataset Preparation

Please refer to [CSDI](https://github.com/ermongroup/CSDI/tree/main) for the dataset preparation.

### Training and inference

Physionet Dataset

```bash
python train_physio.py --testmissingratio [missing ratio] --nsample [number of samples]
```

Air Quality Dataset

```bash
python train_pm25.py --testmissingratio [missing ratio] --nsample [number of samples]
```

### Parameters
For modifying hyperparameters and schedulers, please modify the corresponding parameter in `config/base.yaml`.


### Acknowledgements
A part of the code is based on [CSDI](https://github.com/ermongroup/CSDI/tree/main).