# QuantumEngine

<p align="center">
  <img src="assets/quantumengine-banner.svg" alt="QuantumEngine — exploring the hidden dynamics of financial markets" width="100%">
</p>

<p align="center">
  <strong>An experimental research framework for studying financial-market dynamics through energy-related features, entropy, and probabilistic market states.</strong>
</p>

---

## Overview

QuantumEngine is a Python research project that investigates whether financial-market behavior can be described using measurable properties of price dynamics, an experimental energy representation, entropy, and rolling market-state distributions.

The central research question is:

> Do energy- and entropy-related features contain statistically useful information about future market behavior beyond conventional descriptions such as volatility?

This is an open research question, not an established result. QuantumEngine is intended to make hypotheses measurable and testable—not to assume that markets are predictable or that a trading edge exists.

## Research components

### Market dynamics

The feature layer derives basic quantities from OHLCV data:

- **Log return** — the logarithmic change in closing price:
  \`r_t = ln(close_t / close_{t-1})\`
- **Acceleration** — the first difference of log returns:
  \`a_t = r_t - r_{t-1}\`
- **Rolling volatility** — the rolling standard deviation of log returns.
- **Volume** — the observed volume supplied in the OHLCV data.

### Experimental market energy

The current implementation defines an experimental composite measure:

\[
E_t = \alpha v_t^2 + \beta a_t^2 + \gamma \sigma_t V_t
\]

where \`v_t\` is log return, \`a_t\` is acceleration, \`\sigma_t\` is rolling volatility, \`V_t\` is volume, and \`\alpha, \beta, \gamma\` are configurable coefficients.

This expression is a **modeling hypothesis used by the project**, not a physical energy law. Its terms have different units and scales; the coefficients and normalization are therefore part of the modeling design and must be evaluated empirically.

The normalization module estimates separate quantile-based scales for the squared-return, squared-acceleration, and volatility-volume components. It applies logarithmic compression:

\[
x_{norm} = \ln\left(1 + \frac{x}{P_q(x)}\right)
\]

The default quantile is 0.95. In the unified pipeline, these scales are fitted on the training slice and then reused when transforming later data.

### Binary directional entropy

The binary entropy component estimates the rolling proportions of positive and non-positive returns and calculates normalized Shannon entropy:

\[
H_2 = \frac{-p_{up}\ln(p_{up}) - p_{down}\ln(p_{down})}{\ln 2}
\]

The result is between 0 and 1. Higher values indicate a more balanced up/down distribution within the rolling window; lower values indicate a more one-sided distribution. This is a description of the observed window, not a forecast.

### Five-state entropy

A separate component classifies normalized returns into five states:

- \`strong_down\`
- \`down\`
- \`neutral\`
- \`up\`
- \`strong_up\`

It estimates rolling state frequencies and calculates normalized Shannon entropy over that five-state distribution. The classification uses return z-scores and configurable thresholds.

### Future-return targets

For a configurable horizon \`h\`, the target module constructs:

\[
R_{t,h} = \ln\left(\frac{close_{t+h}}{close_t}\right)
\]

It also assigns a direction label based on the sign of the future return. These are labels for historical research and model evaluation; they are not available at inference time.

## Pipeline and validation design

The unified \`FeatureTargetPipeline\` combines the feature components and supports:

1. OHLCV schema, timestamp, and value validation.
2. Construction of price-dynamics features.
3. Optional binary and five-state entropy features.
4. Experimental energy calculation and train-fitted normalization.
5. Training-dataset creation with future-return targets.
6. Inference-dataset creation without future targets.
7. Optional historical context when transforming sequential data.

The pipeline requires an explicit \`train_only=True\` when fitting normalization scales. This makes the intended training-only fit explicit; callers remain responsible for passing a genuine training slice.

The temporal splitter creates ordered train/validation folds and can include a test segment. It supports a configurable embargo and requires the embargo to be at least as large as the target horizon.

These safeguards reduce particular risks of look-ahead leakage, but they do not by themselves prove that every research experiment is free of leakage or overfitting.

## Repository structure

The main implementation lives in \`src/\`:

| Module | Purpose |
|---|---|
| \`data_loader.py\` | Load OHLCV market data |
| \`data_audit.py\` | Validate market-data integrity |
| \`market_features.py\` | Calculate returns, acceleration, and volatility |
| \`market_energy.py\` | Calculate the experimental energy measure |
| \`energy_normalization.py\` | Fit scales and normalize energy components |
| \`market_entropy.py\` | Calculate binary directional entropy |
| \`state_entropy.py\` | Classify five market states and calculate state entropy |
| \`market_state.py\` | Build market-state representations |
| \`forecast_target.py\` | Construct future-return targets |
| \`datasets.py\` | Prepare research datasets |
| \`splitting.py\` | Create temporal walk-forward splits |
| \`pipeline.py\` | Combine features, targets, and transformations |

The repository root also contains standalone \`inspect_*.py\` scripts for data inspection and research experiments, including energy-component comparisons, entropy analysis, ablation, and predictive-relationship investigations.

## Current research status

The codebase contains the implementation and test coverage for the components described above. The development environment previously reported **99 passing tests**.

That test result is a software-quality checkpoint, not evidence that the research hypotheses are true. The repository does not currently present a consolidated, independently validated result demonstrating predictive superiority or trading profitability.

Key questions remain open:

- Do energy-related features add information beyond volatility and simpler baselines?
- Which components, if any, contribute robust incremental information?
- Does entropy help characterize changing market conditions?
- Do observed relationships persist across different time periods and market regimes?
- Do they survive properly separated out-of-sample evaluation and robustness checks?

Negative or inconclusive results are valid outcomes of this research.

## Installation

Clone the repository:

\`\`\`bash
git clone https://github.com/ArtyomTJ/QuantumEngine.git
cd QuantumEngine
\`\`\`

Create and activate a virtual environment (Windows PowerShell):

\`\`\`powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
\`\`\`

Install dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Tests

Run the automated test suite:

\`\`\`bash
pytest -q
\`\`\`

## Market data

Historical market data is not included in this public repository. The research work has used Binance BTCUSDT one-minute OHLCV data.

Some local research scripts expect a CSV at:

\`data/raw/BTCUSDT-1m-2026-08.csv\`

Obtain the dataset separately and place it at the expected path before running scripts that depend on it. The dataset is excluded from the repository; no market-data file is distributed here.

## Research principles

QuantumEngine distinguishes between **measurement, interpretation, and evidence**:

- A feature is a constructed measurement, not automatically an explanation.
- Entropy describes a distribution; it does not by itself predict what happens next.
- Correlation does not establish causation or a reliable forecasting mechanism.
- Any claimed predictive value must be compared with appropriate baselines and tested out of sample.

The long-term aim is to build a reproducible framework for studying the relationship between market activity, uncertainty, state distributions, and future outcomes.

## Disclaimer

QuantumEngine is an experimental research project. It is not financial advice, an investment recommendation, or a validated trading system. No predictive accuracy or profitability is guaranteed.

---

**Author:** Artyom Larin · [GitHub profile](https://github.com/ArtyomTJ) · [Repository](https://github.com/ArtyomTJ/QuantumEngine)
