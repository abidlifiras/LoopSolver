import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

def df_to_transactions(df, delay_thresholds=[30, 60], age_thresholds=[65, 75]):
    rows = []
    for _, r in df.iterrows():
        items = []
        # categories
        items.append(f"urgency_{r['urgency']}")
        items.append(f"complexity_{r['complexity']}")
        # numeric bins
        for t in age_thresholds:
            if r["age"] > t:
                items.append(f"age>{t}")
        for t in delay_thresholds:
            if r["delay"] > t:
                items.append(f"delay>{t}")
        # priority
        if r["priority"] == 1:
            items.append("priority=1")
        else:
            items.append("priority=0")
        rows.append(items)
    # transform to one-hot DataFrame
    all_items = sorted({it for row in rows for it in row})
    trans = []
    for row in rows:
        trans.append({it: (1 if it in row else 0) for it in all_items})
    return pd.DataFrame(trans)

def mine_patterns(df, min_support=0.05, min_confidence=0.6):
    trans = df_to_transactions(df)
    freq = apriori(trans, min_support=min_support, use_colnames=True)
    if freq.empty:
        return [], trans
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    # filter rules where consequent is priority=1
    rules_p1 = rules[rules['consequents'].apply(lambda c: 'priority=1' in c)]
    # make readable rules
    patterns = []
    for _, row in rules_p1.sort_values(by="confidence", ascending=False).iterrows():
        antecedent = ", ".join(sorted(list(row['antecedents'])))
        conf = round(row['confidence'], 2)
        sup = round(row['support'], 2)
        patterns.append({"rule": f"{antecedent} -> priority=1", "support": sup, "confidence": conf})
    return patterns, trans
