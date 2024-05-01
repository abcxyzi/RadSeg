# Radar Segmentation Dataset (RadSeg)

RadSeg is a synthetic radar dataset designed for building semantic segmentation models for radar activity recognition. Unlike existing radio classification datasets that only provide signal-wise annotations for short and isolated IQ sequences, RadSeg provides sample-wise annotations for interleaved radar pulse activities that extend across a long time horizon. This makes RadSeg the first annotated public dataset of its kind for radar activity recognition. This dataset is released to the public under the MIT License.

You can access the conference paper here: [https://arxiv.org/abs/2312.09489](https://arxiv.org/abs/2312.09489)

> Z. Huang, A. Pemasiri, S. Denman, C. Fookes and T. Martin, "Multi-Stage Learning for Radar Pulse Activity Segmentation," ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Korea, Republic of, 2024, pp. 7340-7344, doi: 10.1109/ICASSP48485.2024.10445810.
keywords: {Radar;Speech recognition;Radar countermeasures;Radio communication countermeasures;Task analysis;Speech processing;Signal to noise ratio;Multi-stage learning;activity segmentation;radio signal recognition;deinterleaving;radar dataset},

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

Please refer to the associated conference paper (TBA) for further details on RadSeg.

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

The RadSeg dataset is published together with the conference paper titled [Multi-stage Learning for Radar Pulse Activity Segmentation](https://arxiv.org/abs/2312.09489) at the 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2024). Please cite both the dataset and the conference paper if you find them helpful for your research.

```latex
@inproceedings{huang2024multi,
  title={Multi-Stage Learning for Radar Pulse Activity Segmentation},
  author={Huang, Zi and Pemasiri, Akila and Denman, Simon and Fookes, Clinton and Martin, Terrence},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7340--7344},
  year={2024},
  organization={IEEE}
}
```
