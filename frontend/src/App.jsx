import "./App.css";

import Header from "./components/Header";
import MicButton from "./components/MicButton";
import Waveform from "./components/Waveform";
import TranscriptPanel from "./components/TranscriptPanel";
import AnswerPanel from "./components/AnswerPanel";
import LatencyPanel from "./components/LatencyPanel";

function App() {
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

          <MicButton />

          <Waveform />
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