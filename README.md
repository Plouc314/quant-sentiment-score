# Sentiment Score

Quantitative trading based on https://doi.org/10.3390/electronics12183960

## Getting Started

### Python

Create a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/):
```bash
python -m venv .venv
```

Activate the virtual environment:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip3 install -r requirements.txt
```

Install `sentiment` locally:
```bash
pip3 install -e .
```

### Environment Variables

Create a file `.env` similar to `.env.example`, with values for the environment variables.