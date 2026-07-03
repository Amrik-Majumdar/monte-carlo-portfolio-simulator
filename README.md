# Monte Carlo Portfolio Simulator

This repository contains a Python workflow for modeling portfolio outcomes through Monte Carlo simulation. It is designed to explore how repeated trials can help compare possible returns, downside risk, and uncertainty across investment assumptions.

This project is for technical demonstration and educational analysis. It is not financial advice.

## What This Project Shows

- Monte Carlo simulation as a practical modeling technique
- Python-based numerical analysis workflow
- Portfolio risk and return exploration
- Visualization of simulated outcomes
- A compact script-based project that can be extended into a larger analysis tool

## Repository Structure

```text
.
├── monteCarloSim.py    Main simulation script
├── requirements.txt    Python dependencies
├── README.md           Project documentation
└── .gitignore          Local and generated file exclusions
```

## Technical Approach

The simulator uses repeated random trials to estimate a range of possible portfolio outcomes instead of presenting a single deterministic forecast. This makes the uncertainty visible and allows the user to compare the distribution of outcomes under different assumptions.

Typical analysis questions include:

- What range of outcomes appears under the current assumptions?
- How does downside risk change when assumptions change?
- How concentrated or spread out are the simulated results?
- What does the distribution suggest about variability?

## Local Setup

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Usage

```powershell
python monteCarloSim.py
```

Adjust assumptions directly in the script before running a new simulation.

## Public-Safe Data Policy

Do not commit private financial information, brokerage exports, account statements, or personal portfolio records. If sample data is added later, it should be synthetic or clearly public-safe.

## Limitations

- Outputs depend heavily on the assumptions provided by the user.
- Random simulation does not predict future market behavior.
- The project does not account for every real-world cost, tax, liquidity, or behavioral factor.
- Historical or simulated performance should not be treated as investment guidance.

## Future Improvements

- Move assumptions into a config file or command-line interface.
- Add input validation and clearer error handling.
- Add synthetic sample scenarios.
- Add tests for core calculation functions.
- Separate plotting and simulation logic into reusable modules.
