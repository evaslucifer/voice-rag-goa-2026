class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.targetSampleRate = 16000;
    this.inputSampleRate = sampleRate;
  }

  process(inputs) {
    const input = inputs[0];

    if (!input || input.length === 0) {
      return true;
    }

    const inputData = input[0];

    if (!inputData || inputData.length === 0) {
      return true;
    }

    const resampledData = this.resample(
      inputData,
      this.inputSampleRate,
      this.targetSampleRate
    );

    const pcmData = this.float32ToInt16(resampledData);

    this.port.postMessage(pcmData);

    return true;
  }

  resample(inputData, inputSampleRate, targetSampleRate) {
    if (inputSampleRate === targetSampleRate) {
      return inputData;
    }

    const ratio = inputSampleRate / targetSampleRate;
    const outputLength = Math.floor(inputData.length / ratio);

    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const position = i * ratio;
      const index = Math.floor(position);
      const nextIndex = Math.min(index + 1, inputData.length - 1);
      const fraction = position - index;

      output[i] =
        inputData[index] * (1 - fraction) +
        inputData[nextIndex] * fraction;
    }

    return output;
  }

  float32ToInt16(float32Array) {
    const int16Array = new Int16Array(float32Array.length);

    for (let i = 0; i < float32Array.length; i++) {
      const sample = Math.max(-1, Math.min(1, float32Array[i]));

      int16Array[i] =
        sample < 0
          ? sample * 0x8000
          : sample * 0x7fff;
    }

    return int16Array;
  }
}

registerProcessor("audio-processor", AudioProcessor);