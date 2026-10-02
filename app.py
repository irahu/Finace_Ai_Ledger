import os
from datetime import datetime, date
from collections import defaultdict

import pandas as pd
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)

from models import db, User, Transaction, Budget, CATEGORIES
from ml_engine import categorizer, forecast_next_month, detect_anomalies

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'finance.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

app.jinja_env.filters["abs"] = abs

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def user_transactions_df():
    """All of the current user's transactions as a pandas DataFrame."""
    rows = Transaction.query.filter_by(user_id=current_user.id).all()
    if not rows:
        return pd.DataFrame(columns=["id", "description", "amount", "category", "date"])
    return pd.DataFrame([{
        "id": t.id, "description": t.description, "amount": t.amount,
        "category": t.category, "date": t.date,
    } for t in rows])


def retrain_categorizer_for_user():
    """Blend this user's manually-labeled transactions into the model."""
    labeled = (
        Transaction.query
        .filter_by(user_id=current_user.id, category_source="manual")
        .all()
    )
    if len(labeled) >= 5:
        pairs = [(t.description, t.category) for t in labeled]
        categorizer.retrain_with_user_data(pairs)


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("signup.html")
        if User.query.filter_by(username=username).first():
            flash("That username is already taken.", "error")
            return render_template("signup.html")
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("dashboard"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.route("/")
@login_required
def dashboard():
    df = user_transactions_df()

    total_income = float(df[df["amount"] > 0]["amount"].sum()) if not df.empty else 0.0
    total_expenses = float(df[df["amount"] < 0]["amount"].sum()) if not df.empty else 0.0
    net_balance = total_income + total_expenses

    category_totals = defaultdict(float)
    if not df.empty:
        spend = df[df["amount"] < 0]
        grouped = spend.groupby("category")["amount"].sum().abs()
        for cat, val in grouped.items():
            category_totals[cat] = round(float(val), 2)

    forecast = forecast_next_month(df) if not df.empty else None

    recent = (
        Transaction.query.filter_by(user_id=current_user.id)
        .order_by(Transaction.date.desc()).limit(6).all()
    )

    budgets = Budget.query.filter_by(user_id=current_user.id).all()
    budget_status = []
    this_month = date.today().replace(day=1)
    for b in budgets:
        spent = 0.0
        if not df.empty:
            month_df = df[pd.to_datetime(df["date"]).dt.to_period("M") == pd.Period(this_month, "M")]
            cat_df = month_df[(month_df["category"] == b.category) & (month_df["amount"] < 0)]
            spent = float(cat_df["amount"].sum().__abs__()) if not cat_df.empty else 0.0
        pct = round((spent / b.monthly_limit) * 100, 1) if b.monthly_limit else 0
        budget_status.append({
            "category": b.category, "limit": b.monthly_limit,
            "spent": round(spent, 2), "pct": min(pct, 100), "over": spent > b.monthly_limit,
        })

    return render_template(
        "dashboard.html",
        total_income=round(total_income, 2),
        total_expenses=round(abs(total_expenses), 2),
        net_balance=round(net_balance, 2),
        category_labels=list(category_totals.keys()),
        category_values=list(category_totals.values()),
        forecast=forecast,
        recent=recent,
        budget_status=budget_status,
    )


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

@app.route("/transactions")
@login_required
def transactions():
    df = user_transactions_df()
    anomaly_ids = detect_anomalies(df) if not df.empty else set()

    all_tx = (
        Transaction.query.filter_by(user_id=current_user.id)
        .order_by(Transaction.date.desc()).all()
    )
    # keep the anomaly flag in sync with the latest detection pass
    for t in all_tx:
        flagged = t.id in anomaly_ids
        if t.is_anomaly != flagged:
            t.is_anomaly = flagged
    db.session.commit()

    return render_template("transactions.html", transactions=all_tx, categories=CATEGORIES)


@app.route("/transactions/add", methods=["GET", "POST"])
@login_required
def add_transaction():
    if request.method == "POST":
        description = request.form["description"].strip()
        amount = float(request.form["amount"])
        tx_type = request.form["type"]  # "expense" or "income"
        category = request.form.get("category") or None
        tx_date = datetime.strptime(request.form["date"], "%Y-%m-%d").date()

        if tx_type == "expense":
            amount = -abs(amount)
        else:
            amount = abs(amount)

        category_source = "manual"
        if not category:
            prediction = categorizer.predict(description)
            category = prediction["category"]
            category_source = "ml"

        tx = Transaction(
            user_id=current_user.id, description=description, amount=amount,
            category=category, category_source=category_source, date=tx_date,
        )
        db.session.add(tx)
        db.session.commit()
        retrain_categorizer_for_user()

        flash(f"Transaction added — categorized as '{category}'"
              + (" (AI suggested)" if category_source == "ml" else ""), "success")
        return redirect(url_for("transactions"))

    return render_template("add_transaction.html", categories=CATEGORIES, today=date.today().isoformat())


@app.route("/transactions/<int:tx_id>/delete", methods=["POST"])
@login_required
def delete_transaction(tx_id):
    tx = Transaction.query.filter_by(id=tx_id, user_id=current_user.id).first_or_404()
    db.session.delete(tx)
    db.session.commit()
    flash("Transaction deleted.", "success")
    return redirect(url_for("transactions"))


@app.route("/api/predict-category", methods=["POST"])
@login_required
def predict_category():
    """AJAX endpoint the add-transaction form calls as the user types."""
    description = request.json.get("description", "")
    if not description.strip():
        return jsonify({"category": None, "confidence": 0})
    return jsonify(categorizer.predict(description))


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

@app.route("/budgets", methods=["GET", "POST"])
@login_required
def budgets():
    if request.method == "POST":
        category = request.form["category"]
        limit = float(request.form["monthly_limit"])
        existing = Budget.query.filter_by(user_id=current_user.id, category=category).first()
        if existing:
            existing.monthly_limit = limit
        else:
            db.session.add(Budget(user_id=current_user.id, category=category, monthly_limit=limit))
        db.session.commit()
        flash("Budget saved.", "success")
        return redirect(url_for("budgets"))

    all_budgets = Budget.query.filter_by(user_id=current_user.id).all()
    return render_template("budgets.html", budgets=all_budgets, categories=CATEGORIES)


# ---------------------------------------------------------------------------
# Insights (forecast + anomalies, explained)
# ---------------------------------------------------------------------------

@app.route("/insights")
@login_required
def insights():
    df = user_transactions_df()
    forecast = forecast_next_month(df) if not df.empty else None
    anomaly_ids = detect_anomalies(df) if not df.empty else set()
    anomalies = (
        Transaction.query.filter(Transaction.id.in_(anomaly_ids)).all()
        if anomaly_ids else []
    )
    return render_template("insights.html", forecast=forecast, anomalies=anomalies)


with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
