## Phase 5 Final Configuration

- Horizon: 20 trading days (T+20)
- Threshold: 5 bps
- Feature set: market_only
- Model: Logistic Regression

## Metrics

- Best average ROC-AUC: 0.5828
- 2023 window ROC-AUC: 0.5477
- 2024 window ROC-AUC: 0.6178

## Conclusion

The best robust walk-forward setup is T+20 + 5 bps + market_only + Logistic Regression.
Sentiment-enhanced feature sets did not consistently outperform this baseline across both 2023 and 2024 windows, so no stable incremental sentiment lift is claimed at this stage.
