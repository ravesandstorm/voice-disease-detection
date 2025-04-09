# voice-disease-detection
Minor project, disease detection from patient's voice using CNN and cloud deployment. 

### Log of changes I had to do for deployment 
```
js native mediarecorder mime type doesn't support wav, only webm
librosa only supports wav
1. had to install ffmpeg on machine to make it work, but it isn't suitable for deployment (servers don't have ffmpeg)
2. solution- npm ffmpeg, website side ffmpeg, but npm import issue
3. solution- using cdns:
  i. cdn ffmpeg, chrome doesn't allow due to vulnerability - "SharedArrayBuffer is not available in browser environment due to security restrictions"
  ii. cdn recorder.js, deprecated, no longer supported
  iii. cdn webaudiorecorder, deprecated
  iv. cdn recordrtc, last option, uses ScriptProcessorNode dependency which is deprecated, but works for now.
Direct wav recording on website, no ffmpeg solution for deployment.
Finally, on to deployment.
```
