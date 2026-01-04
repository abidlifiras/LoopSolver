# src/app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from .data_loader import load_raw, preprocess
from .pattern_mining import mine_patterns
from .model import train, load_model, rules_from_tree
from .feedback import add_feedback
from .constraints import evaluate_rule_on_df, cp_test_single_case
import io, os

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
    try:
        model = load_model()
        rules_text = rules_from_tree(model)
    except Exception:
        rules_text = "Model not trained yet."
    return render_template("patterns.html", patterns=pats, tree_rules=rules_text)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    """
    Upload or paste a CSV with new patients (no id required). Expected columns:
    age, urgency, complexity, delay
    """
    if request.method == "POST":
        text = request.form.get("csv_text")
        file = request.files.get("file")
        try:
            if file and file.filename != "":
                df_new = pd.read_csv(file)
            elif text and text.strip() != "":
                df_new = pd.read_csv(io.StringIO(text))
            else:
                flash("Aucun fichier ou texte fourni", "danger")
                return redirect(url_for("upload"))
            # basic preprocessing to ensure columns exist
            expected = {"age", "urgency", "complexity", "delay"}
            if not expected.issubset(set(df_new.columns)):
                flash(f"Colonnes manquantes. Expect: {expected}", "danger")
                return redirect(url_for("upload"))
            # predict using trained model
            model = load_model()
            # prepare features
            df_new_proc = df_new.copy()
            df_new_proc["urgency_cat"] = df_new_proc["urgency"].map({"low":0, "medium":1, "high":2})
            df_new_proc["complexity_cat"] = df_new_proc["complexity"].map({"low":0, "medium":1, "high":2})
            X = df_new_proc[["age", "urgency_cat", "complexity_cat", "delay"]]
            probs = model.predict_proba(X)[:,1]
            preds = model.predict(X)
            df_new_proc["pred_priority"] = preds
            df_new_proc["pred_prob"] = probs
            # order by predicted probability desc
            df_ordered = df_new_proc.sort_values(by="pred_prob", ascending=False)
            # store in session-like way: here we save temp CSV
            tmp_path = "data/tmp_new.csv"
            df_ordered.to_csv(tmp_path, index=False)
            return render_template("predictions.html", table=df_ordered.to_dict(orient="records"))
        except Exception as e:
            flash(str(e), "danger")
            return redirect(url_for("upload"))
    return render_template("upload.html")

@app.route("/add_predictions", methods=["POST"])
def add_predictions():
    """
    Add the temporary predictions (data/tmp_new.csv) to the main dataset as feedback
    with predicted priority as label; then retrain model.
    """
    tmp_path = "data/tmp_new.csv"
    if not os.path.exists(tmp_path):
        flash("Aucune prédiction temporaire trouvée.", "danger")
        return redirect(url_for("index"))
    df_new = pd.read_csv(tmp_path)
    # convert predicted to dataset format and append
    df = load_raw()
    # build rows with id, priority = pred_priority, keep other fields
    for _, r in df_new.iterrows():
        ex = {
            "age": int(r["age"]),
            "urgency": r["urgency"],
            "complexity": r["complexity"],
            "delay": int(r["delay"]),
            "priority": int(r["pred_priority"])
        }
        df = add_feedback(df, ex)
    # retrain
    dfp = preprocess(df)
    train(dfp)
    flash("Prédictions ajoutées et modèle ré-entraîné.", "success")
    return redirect(url_for("patterns"))

@app.route("/propose", methods=["GET", "POST"])
def propose():
    """
    Two actions:
     - Test rule satisfaction (action=test)
     - Submit rule as an example (action=submit) -> append an example or do other action
    """
    if request.method == "POST":
        action = request.form.get("action")
        rule_str = request.form.get("rule")
        if not rule_str:
            flash("Aucune règle fournie.", "danger")
            return redirect(url_for("propose"))
        df = load_raw()
        eval_res = evaluate_rule_on_df(rule_str, df)
        if action == "test":
            # show results without modifying dataset
            flash(f"Antecedent true for {eval_res['count_antecedent']} rows; "
                  f"{eval_res['count_antecedent_and_consequent']} satisfy consequent; "
                  f"{len(eval_res['violations'])} violations.", "info")
            return render_template("propose_rule.html", eval_res=eval_res, last_rule=rule_str)
        elif action == "submit":
            try:
                # create a synthetic example from antecedent items (best-effort)
                example = {}
                antecedents, consequent = None, None
                try:
                    antecedents, consequent = parse_rule(rule_str)
                except:
                    # fallback simple split
                    parts = rule_str.split("->")
                    antecedents = [p.strip() for p in parts[0].split(",")]
                    consequent = parts[1].strip()
                # produce example with default values
                example_row = {"age": 50, "urgency": "low", "complexity": "low", "delay": 5, "priority": 0}
                # try setting attributes from antecedents
                for it in antecedents:
                    if "age" in it and ">" in it:
                        example_row["age"] = int(float(it.split(">")[1]) + 1)
                    if "delay" in it and ">" in it:
                        example_row["delay"] = int(float(it.split(">")[1]) + 1)
                    if it.startswith("urgency_"):
                        example_row["urgency"] = it.split("_",1)[1]
                    if it.startswith("complexity_"):
                        example_row["complexity"] = it.split("_",1)[1]
                # consequent priority
                m = None
                import re
                m = re.match(r".*priority\s*[=]\s*([01])", consequent)
                if m:
                    example_row["priority"] = int(m.group(1))
                df2 = add_feedback(load_raw(), example_row)
                dfp = preprocess(df2)
                train(dfp)
                flash("Règle soumise : un exemple synthétique a été ajouté et le modèle ré-entraîné.", "success")
                return redirect(url_for("patterns"))
            except Exception as e:
                flash(str(e), "danger")
                return redirect(url_for("propose"))
    return render_template("propose_rule.html", eval_res=None, last_rule=None)


from .constraints import parse_rule
