import sys
import augment
import torch
import random
import numpy as np
import torchaudio
from torchaudio.compliance import kaldi

import nlpaug.augmenter.audio as naa

from espnet2.bin.enh_inference import SeparateSpeech

from torchaudio_augmentations import RandomApply, Compose
from vistec_ser.data.features.transform import FilterBank

class Noise(torch.nn.Module):
    def __init__(self, min_snr=0.001, max_snr=1.0):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, au):
        audio, emotion = au['feature'], au['emotion']
        std = torch.std(audio)
        noise_std = random.uniform(self.min_snr * std, self.max_snr * std)

        noise = np.random.normal(0.0, noise_std, size=audio.shape).astype(np.float32)

        return {'feature':audio + noise, 'emotion':emotion}

class Reverb(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        reverberance_min=0,
        reverberance_max=100,
        dumping_factor_min=0,
        dumping_factor_max=100,
        room_size_min=0,
        room_size_max=40, #100
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.reverberance_min = reverberance_min
        self.reverberance_max = reverberance_max
        self.dumping_factor_min = dumping_factor_min
        self.dumping_factor_max = dumping_factor_max
        self.room_size_min = room_size_min
        self.room_size_max = room_size_max
        self.src_info = {"rate": self.sample_rate}
        self.target_info = {
            "channels": 1,
            "rate": self.sample_rate,
        }

    def forward(self, au):
        audio, emotion = au['feature'], au['emotion']
        reverberance = torch.randint(
            self.reverberance_min, self.reverberance_max, size=(1,)
        ).item()
        dumping_factor = torch.randint(
            self.dumping_factor_min, self.dumping_factor_max, size=(1,)
        ).item()
        room_size = torch.randint(
            self.room_size_min, self.room_size_max, size=(1,)
        ).item()

        num_channels = audio.shape[0]
        effect_chain = (
            augment.EffectChain()
            .reverb(reverberance, dumping_factor, room_size)
            .channels(num_channels)
        )

        audio = effect_chain.apply(
            audio, src_info=self.src_info, target_info=self.target_info
        )

        return {'feature':audio, 'emotion':emotion}

class VtlpAug_torch(torch.nn.Module):
    def __init__(self, sr=16000, zone=(0.0, 1.0), coverage=1, duration=None, fhi=4800, factor=(0.8, 1.2)):
        """
        VTLP: Change vocal tract
        Reference: https://pdfs.semanticscholar.org/3de0/616eb3cd4554fdf9fd65c9c82f2605a17413.pdf
        :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
          augmentation will be applied in first 20% and last 20% of whole audio.
        :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
          operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
          zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
          augmented.
        :param tuple factor: Input data vocal will be increased (decreased). Augmented value will be picked
          within the range of this tuple value. Vocal will be reduced if value is between 0 and 1.
        :param int fhi: Boundary frequency. Default value is 4800.
        :param str name: Name of this augmenter
        """
        super().__init__()
        self.sr = sr
        self.zone = zone
        self.coverage = coverage
        self.duration = duration
        self.fhi = fhi
        self.factor = factor
        self.aug = naa.VtlpAug(self.sr, zone=self.zone, coverage=self.coverage, fhi=self.fhi, factor=self.factor)

    def forward(self, au):
        audio, emotion = au['feature'], au['emotion']
        if len(list(audio.size())) > 1:
          if audio.size()[0] == 1:
            audio_trans = torch.squeeze(audio).numpy()
            audio_trans = self.aug.augment(audio_trans)
            audio_trans = torch.unsqueeze(torch.tensor(audio_trans), dim=0)
          elif audio.size()[0] > 1:
            audio_trans = audio.numpy()
            audio_trans = torch.stack([torch.tensor(self.aug.augment(a)) for a in audio_trans])
        else:
          audio_trans = audio.numpy()
          audio_trans = self.aug.augment(audio_trans)
          audio_trans = torch.tensor(audio_trans)
        return {'feature':audio_trans, 'emotion':emotion}

class SE_torch(torch.nn.Module):
    def __init__(self, tag = "enh_model_sc", sr=16000):
        super().__init__()
        self.sr =sr
        self.speech_enh = SeparateSpeech(enh_train_config=f"{tag}/exp/enh_train_enh_conv_tasnet_raw/config.yaml",
                                           enh_model_file=f"{tag}/exp/enh_train_enh_conv_tasnet_raw/6epoch.pth",
                                           # for segment-wise process on long speech
                                          #  segment_size=2.4, 
                                          #  hop_size=0.8, 
                                           normalize_segment_scale=False, 
                                           show_progressbar=False,
                                           ref_channel=None,
                                           normalize_output_wav=False,
                                           device="cuda:0",)

    def forward(self, au):
        audio, emotion = au['feature'], au['emotion']
        if len(list(audio.size())) > 1:
          if audio.size()[0] == 1:
            audio_trans = torch.squeeze(audio).numpy()
            audio_trans = self.speech_enh(audio_trans[None, ...], fs=self.sr)
            audio_trans = torch.unsqueeze(torch.tensor(audio_trans[0].squeeze()), dim=0)
          elif audio.size()[0] > 1:
            audio_trans = audio.numpy()
            audio_trans = torch.stack([torch.tensor(self.speech_enh(a[None, ...], fs=self.sr)[0].squeeze()) for a in audio_trans])
        else:
          audio_trans = audio.numpy()
          audio_trans = self.speech_enh(audio_trans[None, ...], fs=self.sr)
          audio_trans = torch.tensor(audio_trans[0].squeeze())
        return {'feature':audio_trans, 'emotion':emotion}


def example_transform():
    """this function for testing to transform dataset class based torch framework"""
    audio, sample_rate = torchaudio.backend.sox_io_backend.load(audio_path)
    audio = torch.unsqueeze(audio.mean(dim=0), dim=0) # convert to mono
    resample = True
    if resample:
        audio = kaldi.resample_waveform(audio, orig_freq=sample_rate, new_freq=16000)
    transform = Compose([RandomApply([SE_torch()], p=1.0),
                            FilterBank(frame_length=50,
                                      frame_shift=10,
                                      num_mel_bins=60)])
    audio_transform = transform({'feature': audio, 'emotion': 0})
    print('Successful! Transformed Audio:', audio_transform['feature'].shape)
    
    transform_vtlp = Compose([RandomApply([SE_torch()], p=1.0),
                              VtlpAug_torch(sr=16000), 
                            FilterBank(frame_length=50,
                                      frame_shift=10,
                                      num_mel_bins=60)])
    audio_transform_vtlp = transform_vtlp({'feature': audio, 'emotion': 0})
    print('Successful! Transformed Audio with VTLP:', audio_transform_vtlp['feature'].shape) 

example_transform()
    
    
