import json
import joblib
import pandas as pd

MODEL_PATH = "models/pga_top10_model.joblib"
META_PATH  = "models/meta.json"
DATA_PATH  = "data/ASA-All-PGA-Raw-Data-Tourn-Level.csv"


def main():
    model = joblib.load(MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    expected_cols = meta["feature_columns"]
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    if "Finish_clean" not in df.columns:
        finish_clean = (
            df["Finish"]
            .astype(str)
            .str.replace("T", "", regex=False)
        )
        df["Finish_clean"] = pd.to_numeric(finish_clean, errors="coerce")
    top10_row = df[df["Finish_clean"] <= 10].iloc[0]
    base_row = top10_row.to_dict()
    sample = {col: base_row.get(col, 0) for col in expected_cols}

    # Strokes Gained (valores positivos = mejor)
    if "sg_total" in sample:
        sample["sg_total"] = 4
    if "sg_t2g" in sample:
        sample["sg_t2g"] = 3
    if "sg_app" in sample:
        sample["sg_app"] = 1
    if "sg_ott" in sample:
        sample["sg_ott"] = 2.6
    if "sg_putt" in sample:
        sample["sg_putt"] = 1.5
    if "sg_arg" in sample:
        sample["sg_arg"] = 4.3

    if "hole_par" in sample:
        sample["hole_par"] = 288
    if "strokes" in sample:
        sample["strokes"] = 280

    if "season" in sample:
        sample["season"] = 2022
    if "no_cut" in sample:
        sample["no_cut"] = 0

    X_new = pd.DataFrame([sample], columns=expected_cols)

    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0][1] if hasattr(model, "predict_proba") else None

    print("\n=== Predicción Top 10 ===")
    print("Predicción (0=No, 1=Sí):", int(pred))
    if proba is not None:
        print("Probabilidad Top 10:", round(float(proba), 4))


if __name__ == "__main__":
    main()