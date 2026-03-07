"""
model_pipeline.py
─────────────────
Trains the KNN heart-disease classifier and saves every pipeline artefact.

Run once before starting the server:
    python model_pipeline.py

Or let main.py call build_and_save_pipeline() automatically on first start.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


def build_and_save_pipeline(df_path: str = "heart.csv") -> bool:
    """
    Full ML pipeline:  Load → Clean → Encode → Split → Scale → Train → Save

    Parameters
    ----------
    df_path : str
        Path to heart.csv. Defaults to 'heart.csv' in the working directory.

    Returns
    -------
    bool  True on success, False if heart.csv is not found.
    """

    # ── Step 1 : Load data ────────────────────────────────────────────────────
    if not os.path.exists(df_path):
        print(f"[ERROR] '{df_path}' not found. "
              "Place heart.csv in the same folder as this script.")
        return False

    df  = pd.read_csv(df_path)
    df1 = df.copy()
    print(f"[INFO]  Loaded {len(df1)} records  |  columns: {list(df1.columns)}")

    # ── Step 2 : Label-encode binary categorical columns ──────────────────────
    #   Sex            : F → 0 , M → 1
    #   ExerciseAngina : N → 0 , Y → 1
    #
    #   Two *separate* encoders — saved separately — so main.py can use the
    #   correct one for each column without ambiguity.
    le_sex    = LabelEncoder()
    le_angina = LabelEncoder()

    df1["Sex"]            = le_sex.fit_transform(df1["Sex"])
    df1["ExerciseAngina"] = le_angina.fit_transform(df1["ExerciseAngina"])

    # ── Step 3 : One-hot encode multi-class categoricals ──────────────────────
    df1 = pd.get_dummies(
        df1,
        columns=["ChestPainType", "RestingECG", "ST_Slope"],
        drop_first=True,      # avoids dummy-variable trap
    )
    df1 = df1.astype(int)     # bool → int so StandardScaler works cleanly

    # ── Step 4 : Impute physiologically impossible zeros ──────────────────────
    #   Some records have Cholesterol = 0 or RestingBP = 0 (data-entry errors).
    #   Replace with mean of non-zero values.
    ch_mean = df1.loc[df1["Cholesterol"] != 0, "Cholesterol"].mean()
    bp_mean = df1.loc[df1["RestingBP"]   != 0, "RestingBP"].mean()

    df1["Cholesterol"] = df["Cholesterol"].replace(0, ch_mean).round(2)
    df1["RestingBP"]   = df["RestingBP"].replace(0, bp_mean).round(2)

    # ── Step 5 : Feature / target split ──────────────────────────────────────
    X = df1.drop("HeartDisease", axis=1)
    y = df1["HeartDisease"]

    print(f"[INFO]  Feature matrix shape : {X.shape}")
    print(f"[INFO]  Features : {list(X.columns)}")

    # ── Step 6 : Train / test split (80 / 20) ────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ── Step 7 : StandardScaler — fit ONLY on X_train ────────────────────────
    #   Fitting on the full dataset would leak test-set statistics into the model.
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)          # transform only — no leakage

    # ── Step 8 : Train KNN (k = 9) ───────────────────────────────────────────
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train_scaled, y_train)

    y_pred   = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"[INFO]  Test accuracy : {accuracy:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    # ── Step 9 : Save all artefacts ───────────────────────────────────────────
    artefacts = {
        "knn_model.pkl"       : model,
        "scaler.pkl"          : scaler,
        "le_sex.pkl"          : le_sex,
        "le_angina.pkl"       : le_angina,        # separate encoder — critical fix
        "feature_columns.pkl" : list(X.columns),
        "categorical_cols.pkl": [
            "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"
        ],
    }

    for filename, obj in artefacts.items():
        with open(filename, "wb") as fh:
            pickle.dump(obj, fh)
        print(f"[SAVED] {filename}")

    print("\n[OK] All artefacts ready. You can now start main.py.\n")
    return True


# ── Entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_and_save_pipeline()
