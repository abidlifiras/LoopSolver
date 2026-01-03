import pandas as pd

def add_feedback(df, example):
    """
    example: dict with keys age, urgency, complexity, delay, priority
    returns new df with appended example
    """
    new_df = df.copy()
    # assign new id
    new_id = new_df['id'].max() + 1 if not new_df.empty else 0
    example_row = {
        "id": new_id,
        "age": example["age"],
        "urgency": example["urgency"],
        "complexity": example["complexity"],
        "delay": example["delay"],
        "priority": example["priority"]
    }
    new_df = pd.concat([new_df, pd.DataFrame([example_row])], ignore_index=True)
    # save back to disk if desired
    new_df.to_csv("data/patients.csv", index=False)
    return new_df
