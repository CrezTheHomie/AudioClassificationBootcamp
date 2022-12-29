# Why this repository

The purpose of this repository is going to be planning out the learning required that I will need as well as practicing the different classifications for the program I'm thinking of building.  
The program I'm thinking of building is going to be a software solution that mixes an audio classifier for recognizing an accent based on an audio stream as well as live feedback building a transcript of speech and showing what that transcript would look like with a different accent.  

Initial learning steps are to follow [Valerio Valardo's Youtube course][the sound of AI]  
[the sound of AI]:https://www.youtube.com/@ValerioVelardoTheSoundofAI

## Steps to take

Seperated out, the steps I need to learn are:
1. Make a transcript of the audio by recognizing human speech.

 I'll need to start by figuring out the steps to take to 'comprehend speech'. HMM?

2. Make an IPA transcript of audio from human speech.
3. Use a model to predict accent based on human speech.

## Notes 

We use the FFT on a wave form to convert a signal from a time domain and get its frequency domain. In audio ML the STFT, or short time Fourier Transform, takes fixed frames (e.g. 2048 samples) and outputs a spectogram. Spectograms reflect time vs frequency vs amplitude (with amplitude being a heatmap). [See lesson: Valerio Velardo](https://youtu.be/m3XbqfIij_Y?t=1004) 

MFCCs capture the timbral and textural aspect of sound. It is our current best way to understand how the human brain understands sound. Audio Wave -> STFT -> Mel spaced Filter bank -> log of Mel Spectogram -> DCT for MFCC [See lesson: Maziar Raissi from CU Boulder](https://www.youtube.com/watch?v=hF72sY70_IQ)

Interesting note: 
> Cepstral is the "inverse" of Spectral