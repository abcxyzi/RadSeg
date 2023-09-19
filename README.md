# Radar Activity Segmentation Dataset (RadSeg)

RadSeg is a synthetic radar dataset designed for building semantic segmentation models for radar activity recognition. Unlike existing radio classification datasets that only provide signal-wise annotations for short and isolated IQ sequences, RadSeg provides sample-wise annotations for interleaved radar pulse activities that extend across an extended time horizon. This makes RadSeg the first annotated public dataset of its kind for radar activity recognition. This dataset is released to the public under the MIT License. The associated conference paper (currently under review) can be previewed here: https://arxiv.org/abs/2306.13105

> Huang, Zi, Akila Pemasiri, Simon Denman, Clinton Fookes, and Terrence Martin. "Multi-stage Learning for Radar Pulse Activity Segmenation" In Proceedings of the 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2024) (under review).

## Dataset Details

RadSeg contains pulsed radar signals at varying signal-to-noise ratios (SNRs) between -20 to 20 dB with a resolution of 0.5 dB. This repository provides the RadSeg dataset which consists of three parts:

- `RadSeg-Train` contains 60,000 radar signals for model training;
- `RadSeg-Valadation` contains 10,000 radar signals for model validation; and
- `RadSeg-Test` contains 10,000 radar signals held out for testing.

This dataset comprises a total of 5 radar signal types, which include: 
- Barker codes, up to a code length of 13;
- Polyphase Barker codes, up to a code length of 13;
- Frank codes, up to a code length of 16;
- Linear frequency-modulated (LFM) pulses; and 
- Coherent unmodulated pulse trains. 

Additional dataset characteristics:
- The sampling rate used in RadSeg is 3.2 MHz. 
- Each radar signal contains 32,768 complex, baseband IQ samples.
- Annotations are provided as channel-wise binary masks where each channel corresponds to a signal type.

Please refer to the conference paper for further details on RadSeg.

## Download Links

The RadSeg datasets can be downloaded from the following links:

### Raw IQ Data

- [`RadSeg-IQ-Train`](https://radseg.s3.amazonaws.com/train/radseg_iq.hdf5) - approx. file size of 29.3 GB
- [`RadSeg-IQ-Validation`](https://radseg.s3.amazonaws.com/validation/radseg_iq.hdf5) - approx. file size of 4.9 GB
- [`RadSeg-IQ-Test`](https://radseg.s3.amazonaws.com/test/radseg_iq.hdf5) - approx. file size of 4.9 GB

### Segmentation Masks (Channel-wise Annotations)

- [`RadSeg-Masks-Train`](https://radseg.s3.amazonaws.com/train/radseg_labels.hdf5) - approx. file size of 87.9 GB
- [`RadSeg-Masks-Validation`](https://radseg.s3.amazonaws.com/validation/radseg_labels.hdf5) - approx. file size of 14.6 GB
- [`RadSeg-Masks-Test`](https://radseg.s3.amazonaws.com/test/radseg_labels.hdf5) - approx. file size of 14.6 GB

### SNR Labels

- [`RadSeg-SNR-Train`](https://radseg.s3.amazonaws.com/train/radseg_snrs.hdf5) - approx. file size of 470.8 KB
- [`RadSeg-SNR-Validation`](https://radseg.s3.amazonaws.com/validation/radseg_snrs.hdf5) - approx. file size of 80.1 KB
- [`RadSeg-SNR-Test`](https://radseg.s3.amazonaws.com/test/radseg_snrs.hdf5) - approx. file size of 80.1 KB

## Citation

RadSeg is published (under review) together with the conference paper titled [Multi-stage Learning for Radar Pulse Activity Segmenation](https://arxiv.org/abs/2306.13105) at the 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2023). Please cite both the dataset and the conference paper if you find them helpful for your research.

```latex
@inproceedings{huang2024radseg,
  title={Multi-stage Learning for Radar Pulse Activity Segmenation},
  author={Huang, Zi and Pemasiri, Akila and Denman, Simon and Fookes, Clinton and Martin, Terrence},
  booktitle={Proceedings of the 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2024) (under review)},
  year={2024},
}
```
