const latencyStages = [
  { key: "stt", label: "STT" },
  { key: "embedding", label: "Embedding" },
  { key: "retrieval", label: "Qdrant" },
  { key: "guardrail", label: "Guardrail" },
  { key: "llm", label: "LLM TTFT" },
];

function LatencyPanel({ latency = {} }) {
  // const total = latency.total ?? null;
  const total = latency.total ?? null;

  const stageValues = latencyStages.map((stage) => latency[stage.key] ?? 0);

  const maxStageLatency = Math.max(...stageValues, 1);

  return (
    <section className="panel latency-panel">
      <div className="panel-header">
        <h3>Latency</h3>
        <span className="latency-total">
          Total {total !== null ? `${total} ms` : "— ms"}
        </span>
      </div>

      <div className="latency-list">
        {latencyStages.map((stage) => {
          const value = latency[stage.key] ?? null;
          const percentage =
            value !== null && total ? ((value / total) * 100).toFixed(1) : null;

          return (
            <div className="latency-row" key={stage.key}>
              <span>{stage.label}</span>

              <div className="latency-bar-container">
                <div
                  className="latency-bar"
                  style={{
                    // width: value ? `${Math.min(value, 200) / 2}%` : "0%",
                    width: value ? `${(value / maxStageLatency) * 100}%` : "0%",
                  }}
                ></div>
              </div>

              {/* <span>{value !== null ? `${value} ms` : "— ms"}</span> */}
              <span>
                {value !== null ? `${value} ms (${percentage}%)` : "— ms"}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default LatencyPanel;
