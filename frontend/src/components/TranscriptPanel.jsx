function TranscriptPanel({ transcript = "" }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Transcript</h3>
        <span className="panel-status">LIVE</span>
      </div>

      <p className="placeholder">
        {transcript || "Your speech transcript will appear here..."}
      </p>
    </section>
  );
}

export default TranscriptPanel;