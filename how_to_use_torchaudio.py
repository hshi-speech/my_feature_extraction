import os

import librosa
from torch.utils import data
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F 
import torch

wav_path = '/CDShare/REVERB_DATA/raw_wsj0_data/data/primary_microphone/si_tr/c02/c02c020w.wav'


noisy_ori, _ = librosa.load(wav_path, sr=16000)   # (104500,)
noisy_mag, _ = librosa.magphase(librosa.stft(noisy_ori, n_fft=512, hop_length=256, win_length=512))    # (257, 409)

waveform, sample_rate = torchaudio.load(wav_path)    # torch.Size([1, 104500])
shape = waveform.size()
waveform = waveform.reshape(-1, shape[-1])

spec_trans = T.Spectrogram(power=None)
mel_trans = T.Spectrogram_trans_MelSpectrogram()
mfcc_trans = T.MelSpectrogram_trans_MFCC()
mfcc_ori = T.MFCC()
inverse = T.inverseSTFT()

spectrogram = spec_trans(waveform)
mag_spec = F.complex_norm(spectrogram,2)
phase_spec = F.angle(spectrogram)

# print(spectrogram.shape)
# print(waveform.size()[-1])
# print(mag_spec.shape)
# print(phase_spec.shape)
# print(spectrogram)
# print(mag_spec)
# print(phase_spec)
# enhanced = librosa.istft(mag_spec.squeeze(0).numpy() * phase_spec.squeeze(0).numpy(), hop_length=200, win_length=400, length=waveform.size()[-1])

mag_spec = mag_spec.pow(1 / 2)
phase_spec = torch.stack([phase_spec.cos(), phase_spec.sin()], dim=-1).to(dtype=mag_spec.dtype, device=mag_spec.device)
mag_spec = mag_spec.unsqueeze(-1).expand_as(phase_spec)
inverse_waveform = inverse(mag_spec * phase_spec, waveform.size()[-1])


mel_spectrogram = mel_trans(F.complex_norm(spectrogram,2))
MFCC = mfcc_trans(mel_spectrogram)
MFCC_ori = mfcc_ori(waveform).squeeze(0)

print(inverse_waveform)
print(inverse_waveform.shape)
librosa.output.write_wav("c02c020w_written.wav", inverse_waveform.squeeze(0).numpy(), sr=16000)
