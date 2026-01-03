import pickle
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = "model_tree.pkl"

def train(df, max_depth=4):
    X = df[["age", "urgency_cat", "complexity_cat", "delay"]]
    y = df["priority"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    # save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model, report

def load_model():
    import pickle
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def rules_from_tree(model, feature_names=["age", "urgency_cat", "complexity_cat", "delay"]):
    return export_text(model, feature_names=feature_names)
