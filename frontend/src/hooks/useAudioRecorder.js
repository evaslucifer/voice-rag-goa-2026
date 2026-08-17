import { useCallback, useRef, useState } from "react";
import { sendVoiceQuery } from "../services/api";
import { pcmChunksToWav } from "../utils/pcmToWav";
// import { float32ToInt16 } from "../utils/audioProcessor";

function useAudioRecorder() {
  const [recordingState, setRecordingState] = useState("idle");
  const [error, setError] = useState(null);

  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  // const processorRef = useRef(null);
  const workletNodeRef = useRef(null);
  const sourceRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const pcmChunksRef = useRef([]);

  const [audioLevel, setAudioLevel] = useState(0);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      pcmChunksRef.current = [];
      

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;
      const analyser = audioContext.createAnalyser();

      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;

      analyserRef.current = analyser;

      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      const updateAudioLevel = () => {
        if (!analyserRef.current) {
          return;
        }

        analyserRef.current.getByteTimeDomainData(dataArray);

        let sum = 0;

        for (let i = 0; i < dataArray.length; i++) {
          const normalizedSample = (dataArray[i] - 128) / 128;
          sum += normalizedSample * normalizedSample;
        }

        const rms = Math.sqrt(sum / dataArray.length);

        setAudioLevel(Math.min(rms * 5, 1));

        animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
      };

      updateAudioLevel();

      await audioContext.audioWorklet.addModule("/audio-processor.js");

      const workletNode = new AudioWorkletNode(audioContext, "audio-processor");

      workletNodeRef.current = workletNode;

      workletNode.port.onmessage = (event) => {
        const pcmData = event.data;

        if (!(pcmData instanceof Int16Array)) {
          console.warn("Unexpected PCM data:", pcmData);
          return;
        }

        pcmChunksRef.current.push(pcmData);

        console.log("PCM chunk:", {
          samples: pcmData.length,
          sampleRate: 16000,
          byteLength: pcmData.byteLength,
        });
      };

      source.connect(workletNode);
      workletNode.connect(audioContext.destination);

      setRecordingState("recording");
    } catch (err) {
      console.error("Microphone error:", err);

      setError("Microphone access was denied or unavailable.");
      setRecordingState("error");
    }
  }, []);

  const stopRecording = useCallback(async () => {
    setRecordingState("processing");

    const chunks = pcmChunksRef.current;

    try {
      if (chunks.length === 0) {
        throw new Error("No audio was captured.");
      }

      const audioBlob = pcmChunksToWav(chunks, 16000);

      console.log("Voice recording ready:", {
        chunks: chunks.length,
        size: audioBlob.size,
        type: audioBlob.type,
      });

      const result = await sendVoiceQuery(audioBlob, "en-IN");

      console.log("Voice RAG response:", result);

      pcmChunksRef.current = [];

      return result;
    } catch (err) {
      console.error("Voice query failed:", err);

      setError(err.message || "Voice query failed.");
      setRecordingState("error");

      return null;
    } finally {
      if (workletNodeRef.current) {
        workletNodeRef.current.disconnect();
        workletNodeRef.current = null;
      }

      if (sourceRef.current) {
        sourceRef.current.disconnect();
        sourceRef.current = null;
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });

        streamRef.current = null;
      }

      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }

      if (analyserRef.current) {
        analyserRef.current.disconnect();
        analyserRef.current = null;
      }

      setAudioLevel(0);
      setRecordingState("idle");
    }
  }, []);

  return {
    recordingState,
    error,
    audioLevel,
    startRecording,
    stopRecording,
  };
}

export default useAudioRecorder;
