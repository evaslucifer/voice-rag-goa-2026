function MicButton({ isRecording = false, onClick }) {
  return (
    <>
      <button
        className={`mic-button ${isRecording ? "recording" : ""}`}
        aria-label={isRecording ? "Stop recording" : "Start recording"}
        onClick={onClick}
      >
        🎙️
      </button>

      <p className="mic-label">
        {isRecording ? "Listening..." : "Click to start speaking"}
      </p>
    </>
  );
}

export default MicButton;