# Battery State-of-Charge Estimation using RNNs

Code to train various models for state-of-charge estimation.

## Data

Included in the `data/` folder of this repository is the [Panasonic 18650PF Li-ion Battery Data](https://data.mendeley.com/datasets/wykht8y7tg/1).

## Code Setup

This code has the following dependencies:

* PyTorch 
* Matplotlib
* Seaborn
* pandas
* mat4py
* scikit-learn

Our RNN script builds on code from [https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network).

## Train an MLP on the SOC Data

```
python mlp.py --window_size 10 --hidden_size 4 --noise_std 0.005
```

## Train an LSTM on the SOC Data

```
python lstm.py --sequence_length 100 --hidden_size 4 --num_layers 4 --noise_std 0.005
```
