# Deep-Learning-for-Time-Series

This repository contains my project for unsupervised anomaly detection on multivariate time series.

The experiment is performed on the **SMD machine-1-1** sequence. The training data is treated as normal, and test labels are used only for final evaluation.

## Methods

I compare two reconstruction-based approaches:

- a convolutional autoencoder trained only on normal windows;
- MOMENT, a pretrained time-series foundation model used zero-shot in reconstruction mode.

Both methods use the same preprocessing, window length, anomaly score and calibration-based thresholding protocol.

## Main result

Both models produce informative anomaly scores, but MOMENT gives a cleaner binary detector after thresholding.

| Method | AUROC-W | AUPRC-W | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Conv. AE | 0.9596 | 0.8366 | 0.2204 | 1.0000 | 0.3612 |
| MOMENT zero-shot | 0.9697 | 0.8391 | 0.8556 | 0.7855 | 0.8190 |

## Files

- `smd_anomaly_detection.ipynb`: notebook used to reproduce the experiments.

## Summary

The autoencoder detects all anomalous windows but produces many false positives. MOMENT is more selective and achieves a much higher F1 score under the same calibration rule.
