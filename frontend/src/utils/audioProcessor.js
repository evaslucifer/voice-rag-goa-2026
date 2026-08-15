export function float32ToInt16(float32Array) {
  const int16Array = new Int16Array(float32Array.length);

  for (let i = 0; i < float32Array.length; i++) {
    const sample = Math.max(-1, Math.min(1, float32Array[i]));

    int16Array[i] = sample < 0
      ? sample * 0x8000
      : sample * 0x7fff;
  }

  return int16Array;
}