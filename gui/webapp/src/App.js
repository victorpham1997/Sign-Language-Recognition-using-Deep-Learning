import logo from './logo.svg';
import './App.css';
import VideoRecorder from 'react-video-recorder'
import { useState } from 'react';

const DOMAIN = 'http://localhost:5000'
function App() {

  let [result, setResult] = useState('Processing...');
  let [isRecording, setIsRecording] = useState(false);
  let [submittedRecording, setSubmittedRecording] = useState(false);

  async function submitVideo(blob) {

    const formData = new FormData();
    formData.append("video", blob);

    let response = await fetch(`${DOMAIN}/identify-sign`, {
      method: 'POST',
      body: formData,
    })

    let text = await response.text();
    setResult(text);
  }

  

  return (
    <div style={{height: '100vh', display: 'flex', flexDirection: 'column'}}>
      <div style={{fontWeight: 600, fontSize: 32, margin: 8, marginLeft: 20, fontFamily: 'montserrat', color: 'tomato'}}>AUTSL Computer Vision - Sign Language Identifier</div>
      <VideoRecorder
        countdownTime={0}
        style={{flex: 1}}
        isOnInitially={true}
        onRecordingComplete={videoBlob => submitVideo(videoBlob)}
        onStartRecording={_ => {
          setIsRecording(true);
        }}
        onStopRecording={_ => {
          setSubmittedRecording(true);
          setIsRecording(false);
        }}
        onStopReplaying={_=>{
          setSubmittedRecording(false);
        }}
      />
      <div style={{textAlign: 'right', fontWeight: 600, fontSize: 32, margin: 8, marginLeft: 20, fontFamily: 'montserrat', color: 'tomato'}}>{isRecording ? 'Recording...' : submittedRecording ? (result === null) ? `Identifying Word...` : `Identified Word: ${result}` : 'Press record to begin'}</div>
    </div>
  );
}

export default App;
