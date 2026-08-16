const stageLabels = {
  embedding: "Embedding",
  retrieval: "Retrieval",
  guardrail: "Guardrail",
  llm_ttft: "LLM TTFT",
  total: "Total",
};

const percentileLabels = [
  { key: "p50", label: "P50" },
  { key: "p70", label: "P70" },
  { key: "p90", label: "P90" },
  { key: "p100", label: "P100" },
];

function BenchmarkPanel({ benchmark = null }) {
  if (!benchmark) {
    return (
      <section className="panel benchmark-panel">
        <div className="panel-header">
          <h3>Benchmark</h3>
          <span className="panel-status">NO DATA</span>
        </div>

        <p className="placeholder">Benchmark results will appear here.</p>
      </section>
    );
  }

  const {
    total_queries_executed,
    unique_queries_count,
    target_p50_ms,
    achieved_p50_ms,
    target_met,
    stage_percentiles_ms = {},
  } = benchmark;

  return (
    <section className="panel benchmark-panel">
      <div className="panel-header">
        <h3>Benchmark</h3>

        <span
          className={`benchmark-status ${
            target_met ? "benchmark-status-success" : ""
          }`}
        >
          {target_met ? "TARGET MET" : "TARGET MISSED"}
        </span>
      </div>

      <div className="benchmark-summary">
        <div className="benchmark-stat">
          <span>Queries</span>
          <strong>{total_queries_executed}</strong>
        </div>

        <div className="benchmark-stat">
          <span>Unique Queries</span>
          <strong>{unique_queries_count}</strong>
        </div>

        <div className="benchmark-stat">
          <span>Target P50</span>
          <strong>{target_p50_ms} ms</strong>
        </div>

        <div className="benchmark-stat">
          <span>Achieved P50</span>
          <strong>{achieved_p50_ms} ms</strong>
        </div>
      </div>

      <div className="benchmark-percentiles">
        <div className="benchmark-section-header">
          <h4>Stage Percentiles</h4>
        </div>

        <div className="benchmark-table">
          <div className="benchmark-row benchmark-row-header">
            <span>Stage</span>

            {percentileLabels.map((percentile) => (
              <span key={percentile.key}>{percentile.label}</span>
            ))}
          </div>

          {Object.entries(stageLabels).map(([key, label]) => {
            const stage = stage_percentiles_ms[key];

            if (!stage) {
              return null;
            }

            return (
              <div className="benchmark-row" key={key}>
                <span>{label}</span>

                {percentileLabels.map((percentile) => (
                  <span key={percentile.key}>{stage[percentile.key]} ms</span>
                ))}
              </div>
            );
          })}
        </div>
      </div>
      <div className="benchmark-query-section">
        <div className="benchmark-section-header">
          <h4>Query Results</h4>
        </div>

        <div className="benchmark-query-list">
          {benchmark.query_details?.map((query, index) => (
            <article
              className="benchmark-query-card"
              key={`${query.language}-${query.query}-${index}`}
            >
              <div className="benchmark-query-header">
                <span className="benchmark-query-number">{index + 1}</span>

                <span className="benchmark-query-language">
                  {query.language.toUpperCase()}
                </span>
              </div>

              <p className="benchmark-query-text">{query.query}</p>

              <div className="benchmark-query-metrics">
                <span>
                  Confidence: {(query.confidence_score * 100).toFixed(0)}%
                </span>

                <span>Citations: {query.citations_count}</span>

                <span>Avg: {query.avg_latency_ms} ms</span>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

export default BenchmarkPanel;
