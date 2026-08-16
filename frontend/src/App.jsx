import "./App.css";
import { useEffect } from "react";
import useAudioRecorder from "./hooks/useAudioRecorder";

import Header from "./components/Header";
import MicButton from "./components/MicButton";
import Waveform from "./components/Waveform";
import TranscriptPanel from "./components/TranscriptPanel";
import AnswerPanel from "./components/AnswerPanel";
import LatencyPanel from "./components/LatencyPanel";
const mockCitations = [
  {
    id: "source-1",
    title: "MSMARCO-XI Document 1",
    text: "This document contains information related to the user's query.",
    score: 0.94,
  },
  {
    id: "source-2",
    title: "MSMARCO-XI Document 2",
    text: "This source provides additional context for the generated answer.",
    score: 0.87,
  },
];
const mockLatency = {
  stt: 120,
  embedding: 30,
  retrieval: 20,
  guardrail: 5,
  llm: 400,
  total: 575,
};

function App() {
  const { recordingState, error, audioLevel, startRecording, stopRecording } =
    useAudioRecorder();

  const handleMicClick = () => {
    console.log("Mic clicked:", recordingState);
    if (recordingState === "recording") {
      stopRecording();
    } else if (recordingState === "idle" || recordingState === "error") {
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
          <MicButton
            isRecording={recordingState === "recording"}
            onClick={handleMicClick}
          />
          <p className="recording-status">
            {recordingState === "idle" && "Click the microphone to start"}
            {recordingState === "recording" && "Listening..."}
            {recordingState === "processing" && "Processing your question..."}
            {recordingState === "error" && "Microphone unavailable"}
          </p>

          {error && <p className="mic-error">{error}</p>}

          <Waveform
            audioLevel={recordingState === "recording" ? audioLevel : 0}
          />
        </section>

        <section className="content-grid">
          <TranscriptPanel />

          <AnswerPanel
            answer="This is a grounded answer generated from the retrieved corpus."
            citations={mockCitations}
          />
        </section>

        <LatencyPanel latency={mockLatency} />
      </main>
    </div>
  );
}

export default App;
