# Self-Attention-based-GANs-for-Video-Summarization
PyTorch implementation of Self-Attention based models, that are built upon SUM-GAN [1] for Unsupervised Video Summarization. The models were trained and tested on the datasets SumMe, TVSum and COGNIMUSE, without any prior ground-truth data. 
| Model | Architecture |
| --- | --- |
| SUM-GAN-AED | Self-attention as the frame selector mechanism |
| SUM-GAN-STD | Transformer as the encoder | 
| SUM-GAN-ST |  Transformer as the encoder & decoder | 
| SUM-GAN-STSED | Transformer Sequence Encoder as the encoder |
| SUM-GAN-SAT | Self-attention as the frame selector and transformer as the encoder & decoder |


## Training

The datasets are split into random splits. In each split 80% of the data is used for training and 20% for testing. For training the model using a single split, run:
```
python train.py --split_index N (with N being the index of the split)
```

## Evaluation
The models are evaluated using a ground-truth summary per video, running the 'check_fscores_summe_with_gts.py' and 'check_fscores_tvsum_with_gts.py' scripts.

The results are shown below:  
| Model | SumMe | TVSum |
| --- | --- | --- |
| SUM-GAN [1] | 38.7 | 50.8 |
| ACGAN [2] | 46.0 | 58.5 |
| Cycle-SUM [3] | 46.8 | 57.6 | 
| SUM-GAN-sl [4] | 46.8 | **65.3** |
| SUM-GAN-AEE [5] | 48.9 | 58.3 | 
| Proposed-B [6] | 58.8 | 63.5 | 
| **SUM-GAN-AED** (ours) | **64.85** | **63.18** |

The code implementation is based upon https://github.com/j-min/Adversarial_Video_Summary.

## Dependencies 
Python 3.7  
PyTorch 1.12.1

### References
[1] Behrooz Mahasseni, Michael Lam και Sinisa Todorovic. Unsupervised Video Summarization With Adversarial LSTM Networks. Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2017.

[2] X. He, Y. Hua, T. Song, Z. Zhang, Z. Xue, R. Ma, N. Robertson, and H. Guan, “Unsupervised video summarization with attentive conditional generative adversarial networks,” in Proceedings of the 27th ACM International Conference on Multimedia, New York, NY,
USA, 2019, MM ’19, p. 2296–2304, Association for Computing Machinery.

[3] Li Yuan, Francis E. H. Tay, Ping Li, Li Zhou, and Jiashi Feng, “Cycle-sum: Cycle-consistent adversarial LSTM networks for unsupervised video summarization,” CoRR, vol. abs/1904.08265, 2019.

[4] E. Apostolidis, A. I. Metsai, E. Adamantidou, V. Mezaris, and I. Patras, “A stepwise, label-based approach for improving the adversarial training in unsupervised video summarization,” in Proceedings of the
1st International Workshop on AI for Smart TV Content Production, Access and Delivery, 2019.

[5] E. Apostolidis, E. Adamantidou, A. I. Metsai, V. Mezaris, and I. Patras, “Unsupervised video summarization via attention-driven adversarial learning,” in
Proceedings of the International Conference on Multimedia Modeling (MMM). 2020, Springer.

[6] Michail Kaseris, Ioannis Mademlis, and Ioannis Pitas, “Exploiting caption diversity for unsupervised video summarization,” in ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 1650–1654.
