const latencyStages = [
  { key: "stt", label: "STT" },
  { key: "embedding", label: "Embedding" },
  { key: "retrieval", label: "Qdrant" },
  { key: "guardrail", label: "Guardrail" },
  { key: "llm", label: "LLM TTFT" },
];

function LatencyPanel({ latency = {} }) {
  const total = latency.total ?? null;

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

          return (
            <div className="latency-row" key={stage.key}>
              <span>{stage.label}</span>

              <div className="latency-bar-container">
                <div
                  className="latency-bar"
                  style={{
                    width: value ? `${Math.min(value, 200) / 2}%` : "0%",
                  }}
                ></div>
              </div>

              <span>{value !== null ? `${value} ms` : "— ms"}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default LatencyPanel;