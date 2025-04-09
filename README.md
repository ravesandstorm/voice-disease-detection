# voice-disease-detection
Minor project, disease detection from patient's voice using CNN and cloud deployment. 

### Log of changes I had to do for deployment 
```
js native mediarecorder no wav, webm
librosa only wav
native ffmpeg works! not suitable for deployment

website side ffmpeg, npm import issue
cdn ffmpeg, chrome doesn't allow due to vulnerability
cdn recorder js, deprecated
cdn webaudiorecorder, deprecated
cdn recordrtc, dependency deprecated, but works for direct wav ! deployed
```
