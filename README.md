# Ledger — AI-Based Finance Management System

A personal finance web app built with **Flask**, **SQLAlchemy (SQLite)**, **pandas**, **numpy**, and **scikit-learn**. It tracks income/expenses, auto-categorizes transactions with a text-classification model, forecasts next month's spending with a regression model, and flags unusual transactions with anomaly detection.

## Features

- **Auth** — signup/login/logout (Flask-Login, hashed passwords)
- **Transaction ledger** — add, view, delete income and expenses
- **AI auto-categorization** — type a description like "starbucks coffee" and the app suggests a category live, via a TF-IDF/Count-vectorizer + Naive Bayes classifier trained on common transaction patterns. The model keeps learning: once you've manually confirmed 5+ of your own transactions, it retrains blending your data with the seed set, so it adapts to your specific merchants.
- **Spending forecast** — numpy linear regression over your monthly expense history projects next month's total, with a confidence band, on the dashboard and the Signals page.
- **Anomaly detection** — flags any expense that's unusually large for its category using a robust median/MAD-based z-score (resistant to the outlier itself skewing the threshold).
- **Budgets** — set a monthly limit per category, see progress bars on the dashboard.
- **Dashboard** — income/expense/net stat cards, spending-by-category doughnut chart, forecast line chart.

## Project structure

```
finance_ai/
  app.py            Flask routes
  models.py         SQLAlchemy models (User, Transaction, Budget)
  ml_engine.py       Categorizer, forecast_next_month, detect_anomalies
  requirements.txt
  templates/         Jinja2 templates
  static/css/style.css
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Visit `http://127.0.0.1:5000`, sign up, and start adding transactions. The SQLite database (`finance.db`) is created automatically on first run.

## How the ML pieces work

**Categorizer** (`ml_engine.Categorizer`) — a scikit-learn `Pipeline` of `CountVectorizer` (word/bigram counts) + `MultinomialNB`. Trained at startup on ~50 seeded (description, category) examples covering common merchants. Every time you save a transaction without picking a category yourself, the model predicts one; every time you *do* pick manually, that example is remembered, and once you have 5+ manual labels the model retrains on seed data + your data combined.

**Forecast** (`ml_engine.forecast_next_month`) — pulls your transactions into a pandas DataFrame, groups expenses by calendar month, and fits a straight line (`numpy.polyfit`, degree 1) through the monthly totals. The next point on that line is the projection; the standard deviation of the fit's residuals becomes the ± confidence band shown in the UI. Needs at least 2 months of expense history.

**Anomaly detection** (`ml_engine.detect_anomalies`) — per category, computes each transaction's *modified z-score* against that category's median and median absolute deviation (MAD) rather than mean/std-dev, because a single big outlier otherwise inflates a plain standard deviation enough to hide itself. Threshold is 3.5, the standard cutoff from Iglewicz & Hoaglin's outlier rule. Needs at least 3 transactions in a category before it will flag anything.

## Notes for extending it

- Swap SQLite for Postgres by changing `SQLALCHEMY_DATABASE_URI` in `app.py`.
- The categorizer is intentionally small/fast (trains in milliseconds); if you want higher accuracy, swap in a larger seed dataset or a transformer-based embedding + classifier.
- `SECRET_KEY` and `DEBUG` should come from environment variables in production — see the `os.environ.get(...)` calls in `app.py`.
