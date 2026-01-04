import re
import pandas as pd
from ortools.sat.python import cp_model

def parse_rule(rule_str):
    """
    Parse "antecedent -> consequent"
    antecedent: comma separated items
    consequent: single item like 'priority=1'
    """
    parts = rule_str.split("->")
    if len(parts) != 2:
        raise ValueError("Rule must contain '->'")
    antecedent = parts[0].strip()
    consequent = parts[1].strip()
    antecedents = [a.strip() for a in antecedent.split(",") if a.strip()]
    return antecedents, consequent

def eval_item_on_row(item, row):
    """
    Evaluate a single item on a pandas Series (row).
    item examples: 'age>65', 'delay>30', 'urgency_high', 'complexity_medium', 'priority=1'
    """
    # numeric comparison
    m = re.match(r"^([a-zA-Z_]+)\s*([<>]=?|==|=)\s*(\d+(\.\d+)?)$", item)
    if m:
        col, op, val, _ = m.groups()
        val = float(val)
        cell = row.get(col)
        if cell is None:
            # also try cat name like urgency_high -> split below
            return False
        try:
            cell_val = float(cell)
        except:
            return False
        if op in (">",):
            return cell_val > val
        if op in ("<",):
            return cell_val < val
        if op in (">=",):
            return cell_val >= val
        if op in ("<=",):
            return cell_val <= val
        if op in ("=", "=="):
            return cell_val == val
    # categorical style like urgency_high or complexity_medium
    if "_" in item:
        col, val = item.split("_", 1)
        if col in row:
            return str(row[col]) == val
        # fallback: if column names are suffixed with '_cat' use mapping:
        # e.g. urgency_high -> urgency_cat==2 (but simpler: compare original strings)
        return False
    # equality like priority=1
    m2 = re.match(r"^([a-zA-Z_]+)\s*=\s*(\d+)$", item)
    if m2:
        col, val = m2.groups()
        return str(int(row.get(col, -999))) == val
    # not understood
    return False

def evaluate_rule_on_df(rule_str, df):
    """
    Returns dictionary:
      {
        "antecedents": [...],
        "consequent": "priority=1",
        "count_antecedent": N,
        "count_antecedent_and_consequent": M,
        "violations": list of row indices where antecedent true and consequent false
      }
    """
    antecedents, consequent = parse_rule(rule_str)
    violations = []
    antecedent_true_idx = []
    antecedent_and_consequent_idx = []
    for idx, row in df.iterrows():
        # check antecedent: all items must be true
        ant_true = all(eval_item_on_row(item, row) for item in antecedents)
        if ant_true:
            antecedent_true_idx.append(int(idx))
            # check consequent
            cons_true = eval_item_on_row(consequent, row)
            if cons_true:
                antecedent_and_consequent_idx.append(int(idx))
            else:
                violations.append(int(idx))
    return {
        "antecedents": antecedents,
        "consequent": consequent,
        "count_antecedent": len(antecedent_true_idx),
        "count_antecedent_and_consequent": len(antecedent_and_consequent_idx),
        "violations": violations
    }

def cp_test_single_case(rule_str, example_dict):
    """
    rule_str: 'delay>60 -> priority=1'
    example_dict: dict with keys age, urgency, complexity, delay, priority (priority maybe unknown)
    This function sets up a mini CP model for the single example and enforces the consequent
    if antecedent true. Returns whether consistent and assigned vars.
    """
    antecedents, consequent = parse_rule(rule_str)
    # For simplicity we only support numeric antecedents of form col > val
    model = cp_model.CpModel()
    # create variables for priority (0/1) if not provided
    priority_val = example_dict.get("priority", None)
    priority = model.NewBoolVar("priority")
    # Evaluate antecedent on example dict (no need CP)
    antecedent_true = all(_eval_item_on_example(item, example_dict) for item in antecedents)
    if antecedent_true:
        # enforce consequent priority == 1
        model.Add(priority == 1)
    # if priority known, add it
    if priority_val is not None:
        model.Add(priority == int(priority_val))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return {"satisfiable": True, "priority": solver.Value(priority)}
    else:
        return {"satisfiable": False}

def _eval_item_on_example(item, ex):
    # allow numeric comparisons col>val
    m = re.match(r"^([a-zA-Z_]+)\s*([<>]=?|==|=)\s*(\d+(\.\d+)?)$", item)
    if m:
        col, op, val, _ = m.groups()
        val = float(val)
        cell = ex.get(col)
        if cell is None:
            return False
        try:
            cell_val = float(cell)
        except:
            return False
        if op == ">":
            return cell_val > val
        if op == "<":
            return cell_val < val
        if op == ">=":
            return cell_val >= val
        if op == "<=":
            return cell_val <= val
        if op in ("=", "=="):
            return cell_val == val
    if "_" in item:
        col, val = item.split("_", 1)
        if col in ex:
            return str(ex[col]) == val
        return False
    m2 = re.match(r"^([a-zA-Z_]+)\s*=\s*(\d+)$", item)
    if m2:
        col, val = m2.groups()
        return str(int(ex.get(col, -999))) == val
    return False
