import "./App.css";

function App() {
  return (
    <div className="app">
      <header className="header">
        <div>
          <p className="eyebrow">HH GOA 2026</p>
          <h1>Voice RAG</h1>
        </div>

        <div className="connection-status">
          <span className="status-dot"></span>
          <span>Connected</span>
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <p className="section-label">VOICE ASSISTANT</p>

          <h2>Ask anything about the corpus</h2>

          <p className="hero-description">
            Speak naturally and get a grounded answer from MSMARCO-XI.
          </p>

          <button className="mic-button" aria-label="Start recording">
            🎙️
          </button>

          <p className="mic-label">Click to start speaking</p>

          <div className="waveform">
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
          </div>
        </section>

        <section className="content-grid">
          <div className="panel">
            <div className="panel-header">
              <h3>Transcript</h3>
              <span className="panel-status">LIVE</span>
            </div>

            <p className="placeholder">
              Your speech transcript will appear here...
            </p>
          </div>

          <div className="panel">
            <div className="panel-header">
              <h3>Answer</h3>
              <span className="panel-status">GROUNDED</span>
            </div>

            <p className="placeholder">
              Your grounded answer will appear here...
            </p>
          </div>
        </section>

        <section className="panel latency-panel">
          <div className="panel-header">
            <h3>Latency</h3>
            <span className="latency-total">Total — ms</span>
          </div>

          <div className="latency-list">
            <LatencyRow label="STT" />
            <LatencyRow label="Embedding" />
            <LatencyRow label="Qdrant" />
            <LatencyRow label="Guardrail" />
            <LatencyRow label="LLM TTFT" />
          </div>
        </section>
      </main>
    </div>
  );
}

function LatencyRow({ label }) {
  return (
    <div className="latency-row">
      <span>{label}</span>

      <div className="latency-bar-container">
        <div className="latency-bar"></div>
      </div>

      <span>— ms</span>
    </div>
  );
}

export default App;