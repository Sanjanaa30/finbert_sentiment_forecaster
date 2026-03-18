import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
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



function formatSentiment(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}


function defaultCalibration() {
  return {
    strong_negative_max: -0.25,
    slight_negative_max: -0.05,
    slight_positive_min: 0.05,
    strong_positive_min: 0.25,
    source: "default",
  };
}

function formatThreshold(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function describeSentiment(score, calibration = defaultCalibration()) {
  if (typeof score !== "number" || Number.isNaN(score)) {
    return {
      label: "Loading",
      explanation: "Waiting for the latest sentiment feed.",
      toneClass: "neutral",
    };
  }

  if (score >= calibration.strong_positive_min) {
    return {
      label: "Strongly Positive",
      explanation: "News tone is clearly optimistic today.",
      toneClass: "positive",
    };
  }
  if (score >= calibration.slight_positive_min) {
    return {
      label: "Slightly Positive",
      explanation: "News tone is mildly optimistic today.",
      toneClass: "positive",
    };
  }
  if (score > calibration.slight_negative_max) {
    return {
      label: "Neutral",
      explanation: "News tone is mixed or balanced today.",
      toneClass: "neutral",
    };
  }
  if (score > calibration.strong_negative_max) {
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


function formatDateTime(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function windowLabel(windowKey) {
  if (windowKey === "1d") return "Today so far";
  if (windowKey === "7d") return "Previous 7 calendar days";
  if (windowKey === "30d") return "Previous 30 calendar days";
  return windowKey || "--";
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
  const [activeHeadlineTab, setActiveHeadlineTab] = useState("positive");

  useEffect(() => {
    let active = true;

    async function loadDashboard() {
      try {
        const [liveOverviewData, trendData, volumeData, summaryData, liveComparisonData] = await Promise.all([
          fetchJson("/dashboard/live-overview?max_headlines=40").catch(() => null),
          fetchJson("/dashboard/live-sentiment-trend").catch(() => ({ rows: [] })),
          fetchJson("/dashboard/headline-volume").catch(() => ({ rows: [] })),
          fetchJson("/dashboard/model-summary").catch(() => null),
          fetchJson("/dashboard/live-vs-history").catch((err) => {
            setLiveComparisonError(String(err.message || err));
            return null;
          }),
        ]);

        if (!active) return;
        setOverview(liveOverviewData);
        setFeedLabel(liveOverviewData?.feed_type || "live_sqlite");
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

  const calibration = useMemo(
    () => overview?.calibration || defaultCalibration(),
    [overview]
  );

  const sentimentDescription = useMemo(
    () => describeSentiment(overview?.sentiment_index, calibration),
    [overview, calibration]
  );

  const groupedLiveHeadlines = useMemo(() => {
    return {
      positive: Array.isArray(overview?.positive_headlines) ? overview.positive_headlines.slice(0, 5) : [],
      neutral: Array.isArray(overview?.neutral_headlines) ? overview.neutral_headlines.slice(0, 5) : [],
      negative: Array.isArray(overview?.negative_headlines) ? overview.negative_headlines.slice(0, 5) : [],
    };
  }, [overview]);

  const sentimentMix = useMemo(() => {
    if (!overview) return null;
    const total = typeof overview.headlines_analyzed === "number" ? overview.headlines_analyzed : 0;
    const positive = typeof overview.positive_count === "number"
      ? overview.positive_count
      : Math.round((overview.positive_share || 0) * total);
    const neutral = typeof overview.neutral_count === "number"
      ? overview.neutral_count
      : Math.round((overview.neutral_share || 0) * total);
    const negative = typeof overview.negative_count === "number"
      ? overview.negative_count
      : Math.max(total - positive - neutral, 0);
    return { positive, neutral, negative };
  }, [overview]);

  const headlineTabs = useMemo(() => {
    return [
      {
        key: "positive",
        label: "Positive",
        count: sentimentMix?.positive ?? "--",
        items: groupedLiveHeadlines.positive,
        className: "positive",
      },
      {
        key: "neutral",
        label: "Neutral",
        count: sentimentMix?.neutral ?? "--",
        items: groupedLiveHeadlines.neutral,
        className: "neutral",
      },
      {
        key: "negative",
        label: "Negative",
        count: sentimentMix?.negative ?? "--",
        items: groupedLiveHeadlines.negative,
        className: "negative",
      },
    ];
  }, [groupedLiveHeadlines, sentimentMix]);

  const activeHeadlineGroup = useMemo(
    () => headlineTabs.find((item) => item.key === activeHeadlineTab) || headlineTabs[0],
    [headlineTabs, activeHeadlineTab]
  );

  const liveWindowBadge = useMemo(() => {
    if (!overview) return null;
    return {
      label: windowLabel(overview.window),
      start: formatDateTime(overview.window_start_local),
      end: formatDateTime(overview.window_end_local),
      coverage: overview.coverage || "--",
      timezone: overview.timezone || "--",
    };
  }, [overview]);

  const sentimentDriverCopy = useMemo(() => {
    if (!overview || !sentimentMix) {
      return {
        summary: "Waiting for the latest headline mix.",
        driver: "Once the live feed loads, this card will explain what is pushing the index up or down.",
      };
    }

    const total = typeof overview.headlines_analyzed === "number" ? overview.headlines_analyzed : 0;
    const largestBucket = [
      { key: "positive", label: "positive", count: sentimentMix.positive || 0 },
      { key: "neutral", label: "neutral", count: sentimentMix.neutral || 0 },
      { key: "negative", label: "negative", count: sentimentMix.negative || 0 },
    ].sort((a, b) => b.count - a.count)[0];

    const summary =
      total > 0
        ? `Built from ${total} live headlines: ${sentimentMix.positive} positive, ${sentimentMix.neutral} neutral, ${sentimentMix.negative} negative.`
        : "No live headlines were available for this window.";

    let driver = "Open the Live Market Headlines tabs below to see the top examples in each sentiment group.";
    if (total > 0) {
      if (largestBucket.key === "neutral") {
        driver =
          "Neutral headlines are the largest group, so the index is being shaped mostly by how strong the positive and negative headlines are.";
      } else if (largestBucket.key === "positive") {
        driver =
          "Positive headlines are the largest group, which is helping lift the sentiment index higher than neutral.";
      } else if (largestBucket.key === "negative") {
        driver =
          "Negative headlines are the largest group, so they are putting the most downward pressure on the sentiment index.";
      }

      if (overview.top_positive_headline?.headline && sentimentDescription.toneClass === "positive") {
        driver = `${driver} Strongest positive example: ${overview.top_positive_headline.headline}`;
      } else if (overview.top_negative_headline?.headline && sentimentDescription.toneClass === "negative") {
        driver = `${driver} Strongest negative example: ${overview.top_negative_headline.headline}`;
      }
    }

    return { summary, driver, dominantLabel: largestBucket.label };
  }, [overview, sentimentMix, sentimentDescription.toneClass]);

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
            <span><strong>{`>= ${formatThreshold(calibration.strong_positive_min)}`}</strong> Strongly Positive</span>
            <span><strong>{`${formatThreshold(calibration.slight_positive_min)} to ${formatThreshold(calibration.strong_positive_min)}`}</strong> Slightly Positive</span>
            <span><strong>{`${formatThreshold(calibration.slight_negative_max)} to ${formatThreshold(calibration.slight_positive_min)}`}</strong> Neutral</span>
            <span><strong>{`${formatThreshold(calibration.strong_negative_max)} to ${formatThreshold(calibration.slight_negative_max)}`}</strong> Slightly Negative</span>
            <span><strong>{`< ${formatThreshold(calibration.strong_negative_max)}`}</strong> Strongly Negative</span>
          </div>
          <div className="metric-context">
            <div className="metric-context-title">Why This Score</div>
            <div className="metric-context-copy">{sentimentDriverCopy.summary}</div>
            <div className="metric-mix-row">
              <span className="metric-mix-pill positive">Positive {sentimentMix?.positive ?? "--"}</span>
              <span className="metric-mix-pill neutral">Neutral {sentimentMix?.neutral ?? "--"}</span>
              <span className="metric-mix-pill negative">Negative {sentimentMix?.negative ?? "--"}</span>
            </div>
            <div className="metric-context-copy">{sentimentDriverCopy.driver}</div>
          </div>
          <div className="badge-row">
            <span className="info-badge">Window: {liveWindowBadge?.label || "--"}</span>
            <span className="info-badge">Coverage: {liveWindowBadge?.coverage || "--"}</span>
          </div>
          <div className="metric-foot">
            {liveWindowBadge ? `${liveWindowBadge.start} to ${liveWindowBadge.end} (${liveWindowBadge.timezone})` : "--"}
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
          <div className="badge-row">
            <span className="info-badge">Window: {liveWindowBadge?.label || "--"}</span>
            <span className="info-badge">Coverage: {liveWindowBadge?.coverage || "--"}</span>
          </div>
          <div className="metric-foot">
            {liveWindowBadge ? `${liveWindowBadge.start} to ${liveWindowBadge.end} (${liveWindowBadge.timezone})` : "--"}
          </div>
          <div className="metric-foot">
            Latest mood: {sentimentDescription.label}
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

            <div className="headline-tab-row">
              {headlineTabs.map((group) => (
                <button
                  key={group.key}
                  type="button"
                  className={`headline-tab ${group.className} ${activeHeadlineTab === group.key ? "active" : ""}`}
                  onClick={() => setActiveHeadlineTab(group.key)}
                >
                  <span className="headline-tab-label">{group.label}</span>
                  <span className="headline-tab-count">{group.count}</span>
                </button>
              ))}
            </div>

            <div className={`headline-group single ${activeHeadlineGroup?.className || "positive"}`}>
              <div className="headline-group-title">
                {activeHeadlineGroup ? `${activeHeadlineGroup.label} (${activeHeadlineGroup.count})` : "Headlines"}
              </div>
              <div className="headline-list compact">
                {activeHeadlineGroup?.items?.length ? activeHeadlineGroup.items.map((item, index) => (
                  <div key={`${activeHeadlineGroup.key}-${item.headline}-${index}`} className="headline-item compact">
                    <div className="headline-rank">{index + 1}</div>
                    <div className="headline-body">
                      <a className="headline-title headline-link" href={item.link || "#"} target="_blank" rel="noreferrer">{item.headline}</a>
                      {item.was_translated && item.original_headline && item.original_headline !== item.headline ? (
                        <div className="headline-translation">
                          Original: {item.original_headline}
                        </div>
                      ) : null}
                      <div className="headline-meta">
                        Source: {item.source || "gdelt"} | Label: {item.display_label || item.label} | Score:{" "}
                        {typeof item.score === "number"
                          ? `${item.score >= 0 ? "+" : ""}${item.score.toFixed(3)}`
                          : "--"}
                        {item.relevance_terms?.length ? ` | Matched: ${Array.isArray(item.relevance_terms) ? item.relevance_terms.join(", ") : item.relevance_terms}` : ""}
                      </div>
                    </div>
                  </div>
                )) : (
                  <div className="metric-foot">No headlines in this sentiment group.</div>
                )}
              </div>
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
          <>
            <div className="badge-row">
              <span className="info-badge">Coverage: {liveComparison?.live?.coverage || overview?.coverage || "--"}</span>
              <span className="info-badge">Timezone: America/New_York (ET)</span>
            </div>
            <div className="metric-foot">
              Primary source: {liveComparison.source_policy.primary_source} | Fallback source: {String(liveComparison.source_policy.fallback_source)} | GDELT status: {liveComparison.source_policy.gdelt_status}
            </div>
          </>
        ) : null}
        <div className="headline-summary-grid">
          {[
            { label: "Today Live", data: liveComparison?.live },
            { label: "Previous 7D", data: liveComparison?.recent_7d },
            { label: "Previous 30D", data: liveComparison?.recent_30d },
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
              <div className="headline-meta">
                Window: {windowLabel(item.data?.window)} | {formatDateTime(item.data?.window_start_local)} to {formatDateTime(item.data?.window_end_local)}
              </div>
              <div className="headline-meta">
                Source: {item.data?.window_source_label || item.data?.source || item.data?.feed_type || "--"}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
