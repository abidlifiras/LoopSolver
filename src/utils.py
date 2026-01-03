import pandas as pd
from .data_loader import load_raw, preprocess
from .pattern_mining import mine_patterns
from .model import train, rules_from_tree

def full_pipeline():
    df = load_raw()
    dfp = preprocess(df)
    patterns, trans = mine_patterns(df)
    model, report = train(dfp)
    rules = rules_from_tree(model)
    return {"df": dfp, "patterns": patterns, "model": model, "report": report, "rules": rules}
