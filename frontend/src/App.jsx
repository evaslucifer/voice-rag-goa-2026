import "./App.css";
import { useEffect } from "react";
import useAudioRecorder from "./hooks/useAudioRecorder";

import Header from "./components/Header";
import MicButton from "./components/MicButton";
import Waveform from "./components/Waveform";
import TranscriptPanel from "./components/TranscriptPanel";
import AnswerPanel from "./components/AnswerPanel";
import LatencyPanel from "./components/LatencyPanel";

function App() {
  const { isRecording, error,audioLevel, startRecording, stopRecording } =
    useAudioRecorder();

  const handleMicClick = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);
  return (
    <div className="app">
      <Header />

      <main className="main-content">
        <section className="hero">
          <p className="section-label">VOICE ASSISTANT</p>
          <h2>Ask anything about the corpus</h2>
          <p className="hero-description">
            Speak naturally and get a grounded answer from MSMARCO-XI.
          </p>
          <MicButton isRecording={isRecording} onClick={handleMicClick} />
          {error && <p className="mic-error">{error}</p>}

          <Waveform audioLevel={audioLevel} />
        </section>

        <section className="content-grid">
          <TranscriptPanel />

          <AnswerPanel />
        </section>

        <LatencyPanel />
      </main>
    </div>
  );
}

export default App;
