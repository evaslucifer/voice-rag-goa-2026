import "./App.css";
import { useEffect, useState } from "react";
import useAudioRecorder from "./hooks/useAudioRecorder";
import { checkBackendHealth, sendTextQuery } from "./services/api";
import Header from "./components/Header";
import MicButton from "./components/MicButton";
import Waveform from "./components/Waveform";
import TranscriptPanel from "./components/TranscriptPanel";
import AnswerPanel from "./components/AnswerPanel";
import LatencyPanel from "./components/LatencyPanel";
import BenchmarkPanel from "./components/BenchmarkPanel";
// const mockCitations = [
//   {
//     id: "source-1",
//     title: "MSMARCO-XI Document 1",
//     text: "This document contains information related to the user's query.",
//     score: 0.94,
//   },
//   {
//     id: "source-2",
//     title: "MSMARCO-XI Document 2",
//     text: "This source provides additional context for the generated answer.",
//     score: 0.87,
//   },
// ];
// const mockLatency = {
//   stt: 120,
//   embedding: 30,
//   retrieval: 20,
//   guardrail: 5,
//   llm: 400,
//   total: 575,
// };

const mockBenchmark = {
  total_queries_executed: 36,
  unique_queries_count: 12,
  target_p50_ms: 200,
  achieved_p50_ms: 0.13,
  target_met: true,

  stage_percentiles_ms: {
    embedding: {
      p50: 0,
      p70: 6.19,
      p90: 7.3,
      p100: 9.4,
    },

    retrieval: {
      p50: 0,
      p70: 0.55,
      p90: 0.72,
      p100: 0.85,
    },

    guardrail: {
      p50: 0,
      p70: 0.08,
      p90: 0.15,
      p100: 0.5,
    },

    llm_ttft: {
      p50: 0,
      p70: 0,
      p90: 3,
      p100: 3,
    },

    total: {
      p50: 0.13,
      p70: 7.6,
      p90: 951.59,
      p100: 5449.6,
    },
  },
};

function App() {
  const { recordingState, error, audioLevel, startRecording, stopRecording } =
    useAudioRecorder();
  const [queryResult, setQueryResult] = useState(null);
  const [queryError, setQueryError] = useState(null);
  const [isQuerying, setIsQuerying] = useState(false);

  const handleTestQuery = async () => {
    try {
      setIsQuerying(true);
      setQueryError(null);

      const result = await sendTextQuery(
        "What is Qdrant vector database used for?",
        "en",
      );

      console.log("RAG response:", result);

      setQueryResult(result);
    } catch (err) {
      console.error("Query failed:", err);
      setQueryError(err.message);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleMicClick = () => {
    console.log("Mic clicked:", recordingState);
    if (recordingState === "recording") {
      stopRecording();
    } else if (recordingState === "idle" || recordingState === "error") {
      startRecording();
    }
  };
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);
  useEffect(() => {
    checkBackendHealth()
      .then((data) => {
        console.log("Backend health:", data);
      })
      .catch((error) => {
        console.error("Backend connection failed:", error);
      });
  }, []);
  return (
    <div className="app">
      <Header />

      <main className="main-content">
        <section className="hero">
          <p className="section-label">VOICE ASSISTANT</p>
          <h2>Ask anything about the corpus</h2>
          <p className="hero-description">
            Speak naturally and get a grounded answer from MSMARCO-XI.
          </p>
          <MicButton
            isRecording={recordingState === "recording"}
            onClick={handleMicClick}
          />
          <button
            type="button"
            className="test-query-button"
            onClick={handleTestQuery}
            disabled={isQuerying}
          >
            {isQuerying ? "Querying..." : "Test Backend Query"}
          </button>
          {queryError && <p className="mic-error">{queryError}</p>}
          <p className="recording-status">
            {recordingState === "idle" && "Click the microphone to start"}
            {recordingState === "recording" && "Listening..."}
            {recordingState === "processing" && "Processing your question..."}
            {recordingState === "error" && "Microphone unavailable"}
          </p>

          {error && <p className="mic-error">{error}</p>}

          <Waveform
            audioLevel={recordingState === "recording" ? audioLevel : 0}
          />
        </section>

        <section className="content-grid">
          <TranscriptPanel
            transcript={queryResult?.transcript || queryResult?.query || ""}
          />

          <AnswerPanel
            answer={queryResult?.answer || ""}
            citations={queryResult?.citations || []}
          />
        </section>

        <LatencyPanel latency={queryResult?.latency_breakdown || {}} />
        <BenchmarkPanel benchmark={mockBenchmark} />
      </main>
    </div>
  );
}

export default App;
