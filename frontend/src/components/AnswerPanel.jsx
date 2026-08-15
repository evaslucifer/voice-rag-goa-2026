function AnswerPanel({ answer = "" }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Answer</h3>
        <span className="panel-status">GROUNDED</span>
      </div>

      <p className="placeholder">
        {answer || "Your grounded answer will appear here..."}
      </p>
    </section>
  );
}

export default AnswerPanel;