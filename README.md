# CHiME4-SER-Conv-TasNet-Speech-Enhancement
This repository is developed for speech enhancement based on Conv-TasNet in speech emotion recognition using ESPNet framework.
For more details, the IEEE conference proceeding is available on [TH-SERSE framework](https://www.researchgate.net/publication/360997197_Real-Time_Thai_Speech_Emotion_Recognition_With_Speech_Enhancement_Using_Time-Domain_Contrastive_Predictive_Coding_and_Conv-Tasnet).

## Testing this code
```python
python transforms.py
```

## Example for using this code
```python
audio, sample_rate = torchaudio.backend.sox_io_backend.load(audio_path)
audio = torch.unsqueeze(audio.mean(dim=0), dim=0) # convert to mono
resample = True
if resample: audio = kaldi.resample_waveform(audio, orig_freq=sample_rate, new_freq=16000)
transform = Compose([RandomApply([SE_torch()], p=1.0), # enhance speech quality by reduce unnecessary information
                      RandomApply([VtlpAug_torch(sr=16000)], p=0.5), # simulate new speaker information by new vocal tract style
                      RandomApply([Reverb(sample_rate=16000)], p=0.2) # optional: traditional environment augmentation
                      RandomApply([Noise()], p=0.2) # optional: traditional environment augmentation
                      FilterBank(frame_length=50,
                                frame_shift=10,
                                num_mel_bins=60)])
audio_transform = transform({'feature': audio, 'emotion': 0})
print('Successful! Transformed Audio:', audio_transform['feature'].shape)
```

## References
```
# main applied
@INPROCEEDINGS{9786444,
  author={Yuenyong, Sumeth and Hnoohom, Narit and Wongpatikaseree, Konlakorn and Singkul, Sattaya},
  booktitle={2022 7th International Conference on Business and Industrial Research (ICBIR)}, 
  title={Real-Time Thai Speech Emotion Recognition With Speech Enhancement Using Time-Domain Contrastive Predictive Coding and Conv-Tasnet}, 
  year={2022},
  volume={},
  number={},
  pages={78-83},
  doi={10.1109/ICBIR54589.2022.9786444}}

# SER environment idea
@article{singkul2022vector,
  title={Vector Learning Representation for Generalized Speech Emotion Recognition},
  author={Singkul, Sattaya and Woraratpanya, Kuntpong},
  journal={Heliyon},
  pages={e09196},
  year={2022},
  publisher={Elsevier}
}

# original pipeline framework
@inproceedings{li2021espnet,
  title={ESPnet-SE: End-to-end speech enhancement and separation toolkit designed for ASR integration},
  author={Li, Chenda and Shi, Jing and Zhang, Wangyou and Subramanian, Aswin Shanmugam and Chang, Xuankai and Kamo, Naoyuki and Hira, Moto and Hayashi, Tomoki and Boeddeker, Christoph and Chen, Zhuo and others},
  booktitle={2021 IEEE Spoken Language Technology Workshop (SLT)},
  pages={785--792},
  year={2021},
  organization={IEEE}
}
```
