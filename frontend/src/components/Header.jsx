function Header() {
  return (
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
  );
}

export default Header;