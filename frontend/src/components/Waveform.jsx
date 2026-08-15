function Waveform() {
  return (
    <div className="waveform" aria-label="Audio waveform">
      {Array.from({ length: 15 }).map((_, index) => (
        <span key={index}></span>
      ))}
    </div>
  );
}

export default Waveform;