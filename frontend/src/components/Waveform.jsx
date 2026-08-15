function Waveform({ audioLevel = 0 }) {
  const bars = Array.from({ length: 15 });

  return (
    <div className="waveform" aria-label="Audio waveform" aria-hidden="true">
      {bars.map((_, index) => {
        const centerDistance = Math.abs(index - 7);
        const positionFactor = 1 - centerDistance / 8;

        const height = 8 + audioLevel * 52 * Math.max(positionFactor, 0.25);

        return (
          <span
            key={index}
            style={{
              height: `${height}px`,
            }}
          />
        );
      })}
    </div>
  );
}

export default Waveform;
