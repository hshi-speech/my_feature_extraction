# my_feature_extraction
I have add some feature extraction function for torchaudio. 
waveform -> spectrogram(original codes) 
spectrogram -> fbank(modified)
fbank -> mfcc(modified)

magnitude spectrogram + phase spectrogram -> waveform (modifed)

They can be easily used in online speech enhancement systems and e2e ASR to do the feature extraction. 
But now, I find only batch_size = 1, and feature frames equally to the original ones can be got the right results. 

Still, I do not put some codes to do the whole feature extraction in this repo. 

Hope everything goes well!
