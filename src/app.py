from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from .data_loader import load_raw, preprocess
from .pattern_mining import mine_patterns
from .model import train, load_model, rules_from_tree
from .feedback import add_feedback
import os
app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = "dev-secret"  

DATA_PATH = "data/patients.csv"

@app.route("/")
def index():
    df = load_raw()
    total = len(df)
    urgent = int(df["priority"].sum())
    return render_template("index.html", total=total, urgent=urgent)

@app.route("/patterns")
def patterns():
    df = load_raw()
    pats, trans = mine_patterns(df)
    return render_template("patterns.html", patterns=pats)


@app.route("/pattern/<int:idx>")
def pattern_detail(idx):
    df = load_raw()
    pats, trans = mine_patterns(df)
    if idx < 0 or idx >= len(pats):
        return redirect(url_for('patterns'))
    pat = pats[idx]
    # attempt to load model and render the decision tree rules
    try:
        model = load_model()
        rules = rules_from_tree(model)
    except Exception:
        rules = "No trained model available."
    return render_template("pattern_detail.html", pattern=pat, rules=rules)

@app.route("/propose", methods=["GET", "POST"])
def propose():
    if request.method == "POST":
        try:
            ex = {
                "age": int(request.form["age"]),
                "urgency": request.form["urgency"],
                "complexity": request.form["complexity"],
                "delay": int(request.form["delay"]),
                "priority": int(request.form["priority"])
            }
            df = load_raw()
            df2 = add_feedback(df, ex)
            # retrain
            dfp = preprocess(df2)
            model, report = train(dfp)
            flash("Exemple ajouté et modèle ré-entraîné.", "success")
            return redirect(url_for("patterns"))
        except Exception as e:
            flash(str(e), "danger")
            return redirect(url_for("propose"))
    return render_template("propose_rule.html")
