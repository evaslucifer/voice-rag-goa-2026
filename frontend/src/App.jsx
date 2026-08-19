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
  const [queryResult, setQueryResult] = useState(null);
  const [textQuery, setTextQuery] = useState("");
  const [language, setLanguage] = useState("en");
  const [isQuerying, setIsQuerying] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);

  const voiceLanguage =
    language === "en"
      ? "en-IN"
      : language === "hi"
        ? "hi-IN"
        : language === "te"
          ? "te-IN"
          : language === "ta"
            ? "ta-IN"
            : language === "mr"
              ? "mr-IN"
              : "bn-IN";
  const { recordingState, error, audioLevel, startRecording, stopRecording } =
    useAudioRecorder(voiceLanguage);

  const handleTextQuery = async (event) => {
    event.preventDefault();

    const cleanQuery = textQuery.trim();

    if (!cleanQuery || isQuerying) {
      return;
    }

    try {
      setIsQuerying(true);
      setQueryResult(null);

      const result = await sendTextQuery(cleanQuery, language);

      setQueryResult(result);
    } catch (err) {
      console.error("Text query failed:", err);
      setQueryResult({
        status: "ERROR",
        answer: err.message || "Unable to process your question.",
        citations: [],
        latency_breakdown: {},
      });
    } finally {
      setIsQuerying(false);
    }
  };

  const handleMicClick = async () => {
    console.log("Mic clicked:", recordingState);

    if (recordingState === "recording") {
      const result = await stopRecording();

      if (result) {
        setQueryResult(result);
      }
    } else if (recordingState === "idle" || recordingState === "error") {
      await startRecording();
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
        setBackendOnline(true);
      })
      .catch((error) => {
        console.error("Backend connection failed:", error);
        setBackendOnline(false);
      });
  }, []);
  return (
    <div className="app">
      <Header backendOnline={backendOnline} />

      <main className="main-content">
        <section className="hero">
          <p className="section-label">VOICE ASSISTANT</p>
          <h2>Ask anything about the corpus</h2>
          <p className="hero-description">
            Speak naturally and get a grounded answer from MSMARCO-XI.
          </p>
          <form className="query-form" onSubmit={handleTextQuery}>
            <div className="query-input-wrapper">
              <input
                type="text"
                value={textQuery}
                onChange={(event) => setTextQuery(event.target.value)}
                placeholder="Ask a question about MSMARCO-XI..."
                disabled={isQuerying || recordingState === "recording"}
              />

              <button type="submit" disabled={!textQuery.trim() || isQuerying}>
                {isQuerying ? "Asking..." : "Ask"}
              </button>
            </div>
          </form>

          <div className="language-selector">
            <label htmlFor="language">Language</label>

            <select
              id="language"
              value={language}
              onChange={(event) => setLanguage(event.target.value)}
              disabled={isQuerying || recordingState === "recording"}
            >
              <option value="en">English</option>
              <option value="hi">Hindi</option>
              <option value="te">Telugu</option>
              <option value="ta">Tamil</option>
              <option value="mr">Marathi</option>
              <option value="bn">Bengali</option>
            </select>
          </div>

          <p className="query-divider">OR SPEAK</p>
          <MicButton
            isRecording={recordingState === "recording"}
            onClick={handleMicClick}
          />

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
            status={queryResult?.status || "IDLE"}
          />
        </section>

        <LatencyPanel latency={queryResult?.latency_breakdown || {}} />
        <BenchmarkPanel benchmark={mockBenchmark} />
      </main>
    </div>
  );
}

export default App;
