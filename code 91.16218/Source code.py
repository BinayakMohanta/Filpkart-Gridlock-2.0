import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # TIMESTAMP FEATURES
    # --------------------------------------------------------

    time_parts = df["timestamp"].astype(str).str.split(":", expand=True)

    df["hour"] = time_parts[0].astype(int)
    df["minute"] = time_parts[1].astype(int)

    df["time_slot"] = (
        df["hour"] * 4 +
        df["minute"] // 15
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # REMOVE ORIGINAL TIMESTAMP
    df.drop(columns=["timestamp"], inplace=True)

    # --------------------------------------------------------
    # GEOHASH HIERARCHY
    # --------------------------------------------------------

    df["geo2"] = df["geohash"].astype(str).str[:2]
    df["geo3"] = df["geohash"].astype(str).str[:3]
    df["geo4"] = df["geohash"].astype(str).str[:4]
    df["geo5"] = df["geohash"].astype(str).str[:5]

    # --------------------------------------------------------
    # DAY FEATURES
    # --------------------------------------------------------

    df["day_sin"] = np.sin(
        2 * np.pi * df["day"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day"] / 7
    )

    # --------------------------------------------------------
    # WEATHER FLAGS
    # --------------------------------------------------------

    df["is_rainy"] = (
        df["Weather"] == "Rainy"
    ).astype(int)

    df["is_foggy"] = (
        df["Weather"] == "Foggy"
    ).astype(int)

    df["is_snowy"] = (
        df["Weather"] == "Snowy"
    ).astype(int)

    # --------------------------------------------------------
    # TEMPERATURE FEATURES
    # --------------------------------------------------------

    df["temp_sq"] = (
        df["Temperature"] ** 2
    )

    # --------------------------------------------------------
    # LANE FEATURES
    # --------------------------------------------------------

    df["lane_sq"] = (
        df["NumberofLanes"] ** 2
    )

    # --------------------------------------------------------
    # INTERACTIONS
    # --------------------------------------------------------

    df["hour_lane"] = (
        df["hour"] *
        df["NumberofLanes"]
    )

    df["temp_lane"] = (
        df["Temperature"] *
        df["NumberofLanes"]
    )

    return df


train = create_features(train)
test = create_features(test)

# ============================================================
# MISSING VALUES
# ============================================================

for col in [
    "RoadType",
    "Weather",
    "LargeVehicles",
    "Landmarks"
]:
    train[col] = train[col].fillna("Unknown")
    test[col] = test[col].fillna("Unknown")

temp_median = train["Temperature"].median()

train["Temperature"] = (
    train["Temperature"]
    .fillna(temp_median)
)

test["Temperature"] = (
    test["Temperature"]
    .fillna(temp_median)
)

# ============================================================
# FREQUENCY ENCODING
# ============================================================

for col in [
    "geohash",
    "geo2",
    "geo3",
    "geo4",
    "geo5"
]:

    freq = train[col].value_counts()

    train[f"{col}_freq"] = (
        train[col]
        .map(freq)
    )

    test[f"{col}_freq"] = (
        test[col]
        .map(freq)
        .fillna(0)
    )

# ============================================================
# TARGET
# ============================================================

y = np.log1p(train["demand"])

X = train.drop(
    columns=["Index", "demand"]
)

X_test = test.drop(
    columns=["Index"]
)

# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "geohash",
    "geo2",
    "geo3",
    "geo4",
    "geo5",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather"
]

cat_idx = [
    X.columns.get_loc(col)
    for col in categorical_features
]

# ============================================================
# CROSS VALIDATION
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof = np.zeros(len(X))

test_predictions = np.zeros(
    len(X_test)
)

for fold, (train_idx, valid_idx) in enumerate(
    kf.split(X)
):

    print(f"\nFold {fold+1}")

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    model = CatBoostRegressor(
        iterations=5000,
        depth=8,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=42,
        verbose=200
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_idx,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=300,
        use_best_model=True
    )

    valid_pred = model.predict(
        X_valid
    )

    oof[valid_idx] = valid_pred

    test_predictions += (
        model.predict(X_test)
        / 5
    )

# ============================================================
# VALIDATION SCORE
# ============================================================

cv_score = r2_score(
    np.expm1(y),
    np.expm1(oof)
)

print("\nCV R2 Score:", cv_score)

# ============================================================
# FINAL PREDICTIONS
# ============================================================

final_predictions = np.expm1(
    test_predictions
)

final_predictions = np.clip(
    final_predictions,
    0,
    1
)

# ============================================================
# SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": final_predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nsubmission.csv generated successfully")
print(submission.head())
