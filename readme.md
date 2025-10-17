## Code for the paper "TFD: Spectrally Guided Time–Frequency Diffusion Model For Time Series Imputation"


### Requirements

```bash
pip install -r requirements.txt
```

### Dataset Preparation

Please refer to [CSDI](https://github.com/ermongroup/CSDI/tree/main) for the dataset preparation.

### Training and inference

Physionet Dataset

```bash
python train_physio_cascade_beta2new.py --testmissingratio [missing ratio] --nsample [number of samples]
```

Air Quality Dataset

```bash
python train_pm25_cascade_beta2new.py --testmissingratio [missing ratio] --nsample [number of samples]
```

### Parameters
For modifying hyperparameters and schedulers, please modify the corresponding parameter in `config/base.yaml`.


### Acknowledgements
A part of the code is based on [CSDI](https://github.com/ermongroup/CSDI/tree/main).