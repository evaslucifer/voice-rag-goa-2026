function Header({ backendOnline }) {
  return (
    <header className="header">
      <div>
        <p className="eyebrow">HH GOA 2026</p>
        <h1>Voice RAG</h1>
      </div>

      <div className="connection-status">
        <span
          className={`status-dot ${backendOnline ? "online" : "offline"}`}
        ></span>

        <span>
          {backendOnline ? "Connected" : "Backend Offline"}
        </span>
      </div>
    </header>
  );
}

export default Header;