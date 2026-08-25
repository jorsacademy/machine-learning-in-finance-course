# Deep Learning in Finance

This section introduces neural networks from first principles and then moves to architectures commonly used with financial data.

## Lessons

1. `01_introduction_to_neural_networks.py`
2. `02_forward_backward_propagation.py`
3. `03_activation_functions.py`
4. `04_loss_functions.py`
5. `05_optimization_algorithms.py`
6. `06_cnn_financial_data.py`
7. `07_rnn_basics.py`
8. `08_lstm_financial_time_series.py`
9. `09_gru_financial_time_series.py`
10. `10_attention_mechanism.py`

## Methodology Notes

Financial deep-learning experiments require stricter validation than standard i.i.d. machine-learning examples. The examples in this section preserve chronological order, fit preprocessing steps on training data only, and define targets using information from future periods only after feature construction.

Neural networks can fit noise extremely well. A low training loss does not imply a useful trading model. Always evaluate out-of-sample performance and compare against simple baselines.

## Installation

The deep-learning examples use TensorFlow/Keras in addition to the packages listed in the repository root.

```bash
pip install -r requirements.txt
```

## Disclaimer

These examples are for educational purposes only and are not investment advice.