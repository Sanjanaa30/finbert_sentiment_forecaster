import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";


const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const API_KEY = import.meta.env.VITE_API_KEY || "finbert-dev-key";


async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }

  return response.json();
}


function moodClass(mood) {
  if (mood === "Positive") return "positive";
  if (mood === "Negative") return "negative";
  return "neutral";
}


function formatSentiment(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}


function describeSentiment(score) {
  if (typeof score !== "number" || Number.isNaN(score)) {
    return {
      label: "Loading",
      explanation: "Waiting for the latest sentiment feed.",
      toneClass: "neutral",
    };
  }

  if (score > 0.25) {
    return {
      label: "Strongly Positive",
      explanation: "News tone is clearly optimistic today.",
      toneClass: "positive",
    };
  }
  if (score > 0.05) {
    return {
      label: "Slightly Positive",
      explanation: "News tone is mildly optimistic today.",
      toneClass: "positive",
    };
  }
  if (score >= -0.05) {
    return {
      label: "Neutral",
      explanation: "News tone is mixed or balanced today.",
      toneClass: "neutral",
    };
  }
  if (score >= -0.25) {
    return {
      label: "Slightly Negative",
      explanation: "News tone is mildly negative today.",
      toneClass: "negative",
    };
  }
  return {
    label: "Strongly Negative",
    explanation: "News tone is clearly pessimistic today.",
    toneClass: "negative",
  };
}


export default function App() {
  const [overview, setOverview] = useState(null);
  const [trendRows, setTrendRows] = useState([]);
  const [volumeRows, setVolumeRows] = useState([]);
  const [modelSummary, setModelSummary] = useState(null);
  const [feedLabel, setFeedLabel] = useState("historical");
  const [liveComparison, setLiveComparison] = useState(null);
  const [liveComparisonError, setLiveComparisonError] = useState("");
  const [headline, setHeadline] = useState("Nvidia raises revenue guidance on strong AI demand");
  const [headlineResult, setHeadlineResult] = useState(null);
  const [headlineError, setHeadlineError] = useState("");
  const [loadingScore, setLoadingScore] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadDashboard() {
      try {
        const liveOverviewPromise = fetchJson("/dashboard/live-overview?max_headlines=40").catch(() => null);
        const [liveOverviewData, historicalOverviewData, trendData, volumeData, summaryData, liveComparisonData] = await Promise.all([
          liveOverviewPromise,
          fetchJson("/dashboard/overview"),
          fetchJson("/dashboard/sentiment-trend"),
          fetchJson("/dashboard/headline-volume"),
          fetchJson("/dashboard/model-summary"),
          fetchJson("/dashboard/live-vs-history").catch((err) => {
            setLiveComparisonError(String(err.message || err));
            return null;
          }),
        ]);

        if (!active) return;
        const chosenOverview = liveOverviewData || historicalOverviewData;
        setOverview(chosenOverview);
        setFeedLabel(chosenOverview?.feed_type || "historical_phase4");
        setTrendRows(trendData.rows || []);
        setVolumeRows(volumeData.rows || []);
        setModelSummary(summaryData);
        setLiveComparison(liveComparisonData);
      } catch (err) {
        if (!active) return;
        setError(err.message);
      }
    }

    loadDashboard();
    return () => {
      active = false;
    };
  }, []);

  async function handleAnalyze() {
    setLoadingScore(true);
    setHeadlineError("");
    try {
      const data = await fetchJson("/score_headline", {
        method: "POST",
        body: JSON.stringify({ headline }),
      });
      setHeadlineResult(data.result);
    } catch (err) {
      setHeadlineError(err.message);
    } finally {
      setLoadingScore(false);
    }
  }

  const latestWindow = useMemo(() => {
    if (!modelSummary?.windows?.length) return null;
    return modelSummary.windows[modelSummary.windows.length - 1];
  }, [modelSummary]);

  const forecastProbability = useMemo(() => {
    if (!latestWindow) return null;
    return Math.round((latestWindow.roc_auc || 0) * 100);
  }, [latestWindow]);

  const trendData = useMemo(
    () =>
      trendRows.map((row) => ({
        ...row,
        shortDate: row.date?.slice(5) || row.date,
      })),
    [trendRows]
  );

  const volumeData = useMemo(
    () =>
      volumeRows.slice(-60).map((row) => ({
        ...row,
        shortDate: row.date?.slice(5) || row.date,
      })),
    [volumeRows]
  );

  const sentimentDescription = useMemo(
    () => describeSentiment(overview?.sentiment_index),
    [overview]
  );

  const liveHeadlines = useMemo(() => {
    if (!Array.isArray(overview?.sample_headlines)) return [];
    return overview.sample_headlines.slice(0, 5);
  }, [overview]);

  function feedLabelText(feedType) {
    if (feedType === "live_gdelt") return "Live GDELT";
    return "Historical";
  }

  return (
    <div className="page-shell">
      <header className="page-header">
        <div />
        <div className="header-copy">
          <h1>Financial Sentiment Dashboard</h1>
          <p>React + FastAPI dashboard with historical sentiment context and a GDELT-backed live news mood snapshot.</p>
        </div>
        <div className="header-meta">Model: FinBERT + Logistic Regression</div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="top-grid">
        <article className="metric-card">
          <div className="card-title">Sentiment Index</div>
          <div className={`metric-value ${sentimentDescription.toneClass}`}>
            {formatSentiment(overview?.sentiment_index)}
          </div>
          <div className={`metric-subtitle tone-${sentimentDescription.toneClass}`}>
            {sentimentDescription.label}
          </div>
          <div className="metric-explainer">{sentimentDescription.explanation}</div>
          <div className="metric-legend">
            <span><strong>{"> 0.25"}</strong> Strongly Positive</span>
            <span><strong>0.05 to 0.25</strong> Slightly Positive</span>
            <span><strong>-0.05 to 0.05</strong> Neutral</span>
            <span><strong>-0.25 to -0.05</strong> Slightly Negative</span>
            <span><strong>{"< -0.25"}</strong> Strongly Negative</span>
          </div>
          <div className="metric-foot">
            Updated: {overview?.latest_update || "--"} | Feed: {feedLabelText(feedLabel)}
          </div>
        </article>

        <article className="metric-card">
          <div className="card-title">Headline Volume</div>
          <div className="metric-value dark">
            {typeof overview?.headlines_analyzed === "number"
              ? overview.headlines_analyzed.toLocaleString()
              : "--"}
          </div>
          <div className="metric-subtitle">Articles Today</div>
          <div className="metric-foot">
            Latest mood: {overview?.market_mood || "--"}
          </div>
        </article>

        <article className="metric-card forecast">
          <div className="card-title light">Market Forecast</div>
          <div className="forecast-row">
            Prediction:
            <span className="forecast-badge">UP</span>
          </div>
          <div className="forecast-row">
            Confidence:
            <strong>{forecastProbability ? `${forecastProbability}%` : "--"}</strong>
          </div>
          <div className="forecast-row">
            Model:
            <strong>{modelSummary?.selected_config?.model || "logreg"}</strong>
          </div>
          <div className="metric-foot light">
            Horizon: {modelSummary?.selected_config?.horizon_days || 20} trading days
          </div>
        </article>
      </section>

      <section className="panel wide">
        <div className="panel-heading">Sentiment &amp; SPY Trend</div>
        <div className="panel-caption">
          Mean sentiment, SPY close, and headline volume aligned on the same timeline.
        </div>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={420}>
            <LineChart data={trendData} margin={{ top: 10, right: 28, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(120,132,155,0.2)" />
              <XAxis dataKey="shortDate" minTickGap={28} stroke="#667086" />
              <YAxis yAxisId="sentiment" stroke="#3b8d43" domain={["auto", "auto"]} />
              <YAxis yAxisId="spy" orientation="right" stroke="#7f8795" domain={["auto", "auto"]} />
              <Tooltip />
              <Bar yAxisId="sentiment" dataKey="headline_volume" fill="rgba(173,179,190,0.55)" />
              <Line
                yAxisId="sentiment"
                type="monotone"
                dataKey="mean_sentiment"
                stroke="#2e8b3d"
                strokeWidth={3}
                dot={false}
              />
              <Line
                yAxisId="spy"
                type="monotone"
                dataKey="spy_close"
                stroke="#8d949f"
                strokeWidth={3}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="bottom-grid">
        <article className="panel">
          <div className="panel-heading">Live Sentiment Analysis</div>
          <div className="panel-caption">
            Score a single financial headline with your FinBERT sentiment model.
          </div>
          <div className="score-input-row">
            <input
              value={headline}
              onChange={(event) => setHeadline(event.target.value)}
              placeholder="Paste a headline here"
            />
            <button onClick={handleAnalyze} disabled={loadingScore}>
              {loadingScore ? "Analyzing..." : "Analyze"}
            </button>
          </div>

          {headlineError ? <div className="inline-error">{headlineError}</div> : null}

          <div className="score-list">
            {["positive", "neutral", "negative"].map((label) => {
              const value = headlineResult?.probabilities?.[label] || 0;
              return (
                <div key={label} className="score-item">
                  <div className={`score-label ${label}`}>
                    {label[0].toUpperCase() + label.slice(1)} {Math.round(value * 100)}%
                  </div>
                  <div className="score-track">
                    <div className={`score-fill ${label}`} style={{ width: `${value * 100}%` }} />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="metric-foot">
            Sentiment Score:{" "}
            <strong>
              {headlineResult ? `${headlineResult.score >= 0 ? "+" : ""}${headlineResult.score.toFixed(2)}` : "--"}
            </strong>
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading">Headline Volume Trend</div>
          <div className="panel-caption">
            Daily market attention intensity from the aggregated sentiment index.
          </div>
          <div className="mini-chart-wrap">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={volumeData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(120,132,155,0.2)" />
                <XAxis dataKey="shortDate" minTickGap={24} stroke="#667086" />
                <YAxis stroke="#667086" />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="headline_volume"
                  stroke="#2a74d8"
                  strokeWidth={3}
                  fill="rgba(44, 115, 216, 0.24)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <section className="panel wide">
        <div className="panel-heading">Live Market Headlines</div>
        <div className="panel-caption">
          Fresh market headlines from the GDELT live/recent-news pipeline, scored by your sentiment model before aggregation.
        </div>

        {feedLabel === "historical_phase4" ? (
          <div className="metric-foot">
            GDELT live feed is not available right now, so the dashboard is showing historical fallback data.
          </div>
        ) : (
          <>
            <div className="headline-summary-grid">
              <div className="headline-summary-card">
                <div className="headline-summary-label">Most Positive</div>
                <div className="headline-summary-text">
                  {overview?.top_positive_headline || "--"}
                </div>
              </div>
              <div className="headline-summary-card negative">
                <div className="headline-summary-label">Most Negative</div>
                <div className="headline-summary-text">
                  {overview?.top_negative_headline || "--"}
                </div>
              </div>
            </div>

            <div className="headline-list">
              {liveHeadlines.map((item, index) => (
                <div key={`${item.headline}-${index}`} className="headline-item">
                  <div className="headline-rank">{index + 1}</div>
                  <div className="headline-body">
                    <div className="headline-title">{item.headline}</div>
                    <div className="headline-meta">
                      Source: {item.source || "rss"} | Label: {item.label} | Score:{" "}
                      {typeof item.score === "number"
                        ? `${item.score >= 0 ? "+" : ""}${item.score.toFixed(2)}`
                        : "--"}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </section>

      <section className="panel wide">
        <div className="panel-heading">Live vs Recent News Sentiment</div>
        <div className="panel-caption">
          Compare today&apos;s live sentiment with GDELT-based recent 7-day and 30-day news windows.
        </div>
        {liveComparisonError ? (
          <div className="inline-error">
            Live comparison data could not be loaded: {liveComparisonError}
          </div>
        ) : null}
        {liveComparison?.source_policy ? (
          <div className="metric-foot">
            Primary source: {liveComparison.source_policy.primary_source} | Fallback source: {String(liveComparison.source_policy.fallback_source)} | GDELT status: {liveComparison.source_policy.gdelt_status}
          </div>
        ) : null}
        <div className="headline-summary-grid">
          {[
            { label: "Today Live", data: liveComparison?.live },
            { label: "Recent 7D", data: liveComparison?.recent_7d },
            { label: "Recent 30D", data: liveComparison?.recent_30d },
          ].map((item) => (
            <div key={item.label} className="headline-summary-card">
              <div className="headline-summary-label">{item.label}</div>
              <div className="headline-summary-text">
                {typeof item.data?.sentiment_index === "number"
                  ? `${item.data.sentiment_index >= 0 ? "+" : ""}${item.data.sentiment_index.toFixed(2)}`
                  : "--"}
              </div>
              <div className="headline-meta">
                Mood: {item.data?.market_mood || "--"} | Headlines: {item.data?.headlines_analyzed ?? "--"}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
