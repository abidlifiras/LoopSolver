import os
import pandas as pd

DATA_PATH = "data/patients.csv"
RADS_PATH = "data/radiologists.csv"
ASSIGN_PATH = "data/assignments.csv"

def load_raw():
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess(df):
    # copy and basic encoding
    df2 = df.copy()
    # encode categorical to integers for ML
    df2["urgency_cat"] = df2["urgency"].map({"low":0, "medium":1, "high":2})
    df2["complexity_cat"] = df2["complexity"].map({"low":0, "medium":1, "high":2})
    return df2


def ensure_patient_columns():
    df = pd.read_csv(DATA_PATH)
    changed = False
    if "radiologist_id" not in df.columns:
        df["radiologist_id"] = -1
        changed = True
    if "required_specialty" not in df.columns:
        df["required_specialty"] = "general"
        changed = True
    if changed:
        df.to_csv(DATA_PATH, index=False)


def load_radiologists():
    if os.path.exists(RADS_PATH):
        return pd.read_csv(RADS_PATH)
    else:
        return pd.DataFrame(columns=["id", "name", "email", "specialty", "available"])


def add_radiologist(name, email, specialty="general", available=1):
    df = load_radiologists()
    if df.empty:
        next_id = 0
    else:
        next_id = int(df["id"].max()) + 1
    new = {"id": next_id, "name": name, "email": email, "specialty": specialty, "available": int(available)}
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(RADS_PATH, index=False)
    return next_id


def find_available_by_specialty(specialty):
    df = load_radiologists()
    if df.empty:
        return pd.DataFrame()
    return df[(df["specialty"] == specialty) & (df["available"] > 0)].copy()


def assign_radiologist(patient_id, radiologist_id, mark_unavailable=True):
    p = pd.read_csv(DATA_PATH)
    if "radiologist_id" not in p.columns or "required_specialty" not in p.columns:
        ensure_patient_columns()
        p = pd.read_csv(DATA_PATH)
    if patient_id not in p["id"].values:
        raise ValueError(f"patient {patient_id} not found")
    p.loc[p["id"] == patient_id, "radiologist_id"] = radiologist_id
    p.to_csv(DATA_PATH, index=False)
    if mark_unavailable:
        r = load_radiologists()
        if radiologist_id in r["id"].values:
            r.loc[r["id"] == radiologist_id, "available"] = 0
            r.to_csv(RADS_PATH, index=False)
    # record assignment to history
    rec = {"patient_id": int(patient_id), "radiologist_id": int(radiologist_id)}
    # append to assignments CSV
    if os.path.exists(ASSIGN_PATH):
        adf = pd.read_csv(ASSIGN_PATH)
        adf = pd.concat([adf, pd.DataFrame([rec])], ignore_index=True)
    else:
        adf = pd.DataFrame([rec])
    adf.to_csv(ASSIGN_PATH, index=False)


def assign_priority_patients(strategy="first_fit"):
    """Assign radiologists to patients with priority==1 based on required_specialty and availability.

    strategy: currently only 'first_fit' implemented (take first available matching radiologist).
    """
    ensure_patient_columns()
    patients = pd.read_csv(DATA_PATH)
    radiologists = load_radiologists()
    pri = patients[patients["priority"] == 1].copy()
    # optional ordering: by delay desc (older first)
    pri = pri.sort_values(by=["delay"], ascending=False)
    assignments = []
    unassigned = []
    for _, row in pri.iterrows():
        pid = int(row["id"])
        req_spec = row.get("required_specialty", "general") if "required_specialty" in row else "general"
        cand = find_available_by_specialty(req_spec)
        if cand.empty and req_spec != "general":
            cand = find_available_by_specialty("general")
        if not cand.empty:
            rad_id = int(cand.iloc[0]["id"])
            assign_radiologist(pid, rad_id)
            assignments.append((pid, rad_id))
        else:
            assignments.append((pid, None))
            unassigned.append((pid, req_spec))
    # return both lists: assigned and unassigned (with requested specialty)
    return {"assigned": assignments, "unassigned": unassigned}
