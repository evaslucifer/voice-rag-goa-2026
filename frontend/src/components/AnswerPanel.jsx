function AnswerPanel({
  answer = "",
  citations = [],
  status = "IDLE",
}) {
  const normalizedStatus = status.toUpperCase();

  const statusLabel = {
    SUCCESS: "GROUNDED",
    REFUSED: "REFUSED",
    PARTIAL: "PARTIAL",
    ERROR: "ERROR",
    IDLE: "WAITING",
  };

  const displayStatus = statusLabel[normalizedStatus] || normalizedStatus;

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Answer</h3>

        <span
          className={`panel-status answer-status-${normalizedStatus.toLowerCase()}`}
        >
          {displayStatus}
        </span>
      </div>

      <p className="answer-text">
        {answer || "Your grounded answer will appear here..."}
      </p>

      {citations.length > 0 && (
        <div className="citations">
          <div className="citations-header">
            <h4>Sources</h4>
            <span>{citations.length}</span>
          </div>

          <div className="citation-list">
            {citations.map((citation, index) => (
              <article
                className="citation-card"
                key={citation.id || index}
              >
                <div className="citation-number">{index + 1}</div>

                <div className="citation-content">
                  <h5>
                    {citation.metadata?.title ||
                      citation.metadata?.source ||
                      `Source ${index + 1}`}
                  </h5>

                  {citation.text && <p>{citation.text}</p>}

                  {citation.score !== undefined && (
                    <span className="citation-score">
                      Relevance: {(citation.score * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              </article>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export default AnswerPanel;