"""
ML engine for the AI Finance Manager.

Three models, kept intentionally lightweight so they train in milliseconds
on a laptop with no GPU:

1. Auto-categorizer   - TF-IDF + Multinomial Naive Bayes text classifier.
                         Trained on a seed dataset of common transaction
                         descriptions, then re-trained on top of the user's
                         own labeled transactions as they accumulate.
2. Spending forecaster - numpy linear regression (least-squares fit) over
                         monthly expense totals, projecting next month.
3. Anomaly detector    - per-category z-score on transaction amounts
                         (numpy), flags anything > 2 standard deviations
                         from that category's mean as unusual.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Auto-categorization
# ---------------------------------------------------------------------------

# Seed training data: (description, category). Covers common merchant/
# description patterns so a brand-new user still gets useful predictions
# before they've logged any transactions of their own.
SEED_DATA = [
    ("walmart grocery run", "Groceries"), ("whole foods market", "Groceries"),
    ("trader joes", "Groceries"), ("local supermarket", "Groceries"),
    ("costco bulk shopping", "Groceries"), ("safeway", "Groceries"),

    ("starbucks coffee", "Dining"), ("mcdonalds", "Dining"),
    ("chipotle lunch", "Dining"), ("dominos pizza", "Dining"),
    ("restaurant dinner", "Dining"), ("coffee shop", "Dining"),
    ("doordash order", "Dining"), ("uber eats", "Dining"),

    ("uber ride", "Transport"), ("lyft ride", "Transport"),
    ("gas station fill up", "Transport"), ("shell gas", "Transport"),
    ("parking fee", "Transport"), ("metro card refill", "Transport"),
    ("car service", "Transport"), ("toll payment", "Transport"),

    ("electricity bill", "Utilities"), ("water bill", "Utilities"),
    ("internet bill", "Utilities"), ("phone bill", "Utilities"),
    ("gas utility bill", "Utilities"), ("wifi subscription", "Utilities"),

    ("monthly rent payment", "Rent"), ("apartment rent", "Rent"),
    ("landlord payment", "Rent"),

    ("netflix subscription", "Entertainment"), ("spotify premium", "Entertainment"),
    ("movie tickets", "Entertainment"), ("concert tickets", "Entertainment"),
    ("video game purchase", "Entertainment"), ("streaming service", "Entertainment"),

    ("amazon purchase", "Shopping"), ("clothing store", "Shopping"),
    ("online shopping", "Shopping"), ("electronics store", "Shopping"),
    ("shoes purchase", "Shopping"), ("target shopping", "Shopping"),

    ("pharmacy purchase", "Health"), ("doctor visit copay", "Health"),
    ("gym membership", "Health"), ("dental checkup", "Health"),
    ("health insurance premium", "Health"), ("vitamins purchase", "Health"),

    ("salary deposit", "Income"), ("paycheck", "Income"),
    ("freelance payment received", "Income"), ("refund received", "Income"),
    ("bonus payment", "Income"), ("interest earned", "Income"),

    ("miscellaneous purchase", "Other"), ("atm withdrawal", "Other"),
    ("bank fee", "Other"), ("donation", "Other"),
]


class Categorizer:
    def __init__(self):
        self.pipeline = Pipeline([
            ("vec", CountVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", MultinomialNB(alpha=0.1)),
        ])
        self._fit(SEED_DATA)

    def _fit(self, rows):
        descriptions = [r[0] for r in rows]
        labels = [r[1] for r in rows]
        self.pipeline.fit(descriptions, labels)

    def retrain_with_user_data(self, user_transactions):
        """
        user_transactions: list of (description, category) tuples the user
        has manually confirmed. Blending these with the seed set lets the
        model adapt to a person's specific merchants over time without
        losing the general-purpose seed knowledge.
        """
        combined = SEED_DATA + list(user_transactions)
        self._fit(combined)

    def predict(self, description):
        proba = self.pipeline.predict_proba([description])[0]
        classes = self.pipeline.classes_
        best_idx = int(np.argmax(proba))
        return {
            "category": classes[best_idx],
            "confidence": round(float(proba[best_idx]), 3),
        }


# ---------------------------------------------------------------------------
# 2. Spending forecast
# ---------------------------------------------------------------------------

def forecast_next_month(transactions_df):
    """
    transactions_df: DataFrame with columns ['date', 'amount'].
    Returns a dict with the historical monthly series and a next-month
    projection with a simple confidence band, or None if there isn't
    enough history yet (need at least 2 distinct months).
    """
    if transactions_df.empty:
        return None

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    expenses = df[df["amount"] < 0].copy()
    if expenses.empty:
        return None

    expenses["month"] = expenses["date"].dt.to_period("M")
    monthly = expenses.groupby("month")["amount"].sum().abs().sort_index()

    if len(monthly) < 2:
        return None

    y = monthly.values.astype(float)
    x = np.arange(len(y))

    # Least-squares linear fit: y = m*x + b
    m, b = np.polyfit(x, y, 1)
    next_x = len(y)
    prediction = float(m * next_x + b)
    prediction = max(prediction, 0.0)  # spending can't be negative

    residuals = y - (m * x + b)
    std_error = float(np.std(residuals)) if len(residuals) > 1 else 0.0

    return {
        "months": [str(p) for p in monthly.index],
        "actual": [round(v, 2) for v in y.tolist()],
        "predicted_next_month": round(prediction, 2),
        "confidence_band": round(std_error, 2),
        "trend": "rising" if m > 1 else ("falling" if m < -1 else "stable"),
    }


# ---------------------------------------------------------------------------
# 3. Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(transactions_df, z_threshold=3.5):
    """
    transactions_df: DataFrame with columns ['id', 'category', 'amount'].
    Flags expense transactions whose absolute amount is unusual relative
    to that category's typical spend, using a robust (median / MAD-based)
    z-score rather than mean/std-dev — a single large outlier inflates a
    plain standard deviation enough to hide itself, so this uses the
    modified z-score (Iglewicz & Hoaglin), which stays robust with the
    small samples a personal finance app actually has.
    Returns a set of transaction ids flagged as anomalous.
    """
    if transactions_df.empty:
        return set()

    df = transactions_df.copy()
    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs()

    anomaly_ids = set()
    for category, group in expenses.groupby("category"):
        if len(group) < 3:
            continue  # not enough data to judge what's "normal" yet
        median = group["abs_amount"].median()
        mad = (group["abs_amount"] - median).abs().median()
        if mad == 0 or np.isnan(mad):
            continue
        modified_z = 0.6745 * (group["abs_amount"] - median) / mad
        flagged = group.loc[modified_z.abs() > z_threshold, "id"]
        anomaly_ids.update(flagged.tolist())

    return anomaly_ids


# Module-level singleton so the model isn't retrained on every request
categorizer = Categorizer()
