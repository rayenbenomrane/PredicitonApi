import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


df = pd.read_csv(r"D:\telechargements\employee_attrition_1000.csv", sep=',')

df = df.drop(columns=['Employee_ID'],errors='ignore')
df.columns = df.columns.str.strip()
print(df.columns.tolist())
df['Attrition'] = df['Attrition'].map({"Yes": 1, "No": 0})

X = df.drop(columns=['Attrition'])
y = df["Attrition"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 3. PREPROCESSING (FIX)
# =========================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# =========================
# 4. MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

# SMOTE must be applied BEFORE model
pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", model)
])

# =========================
# 5. TRAIN
# =========================
pipeline.fit(X_train, y_train)

# =========================
# 6. TEST
# =========================
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


with open("attrition_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved successfully!")


def predict_employee(employee: dict) -> dict:
    with open("attrition_model.pkl", "rb") as f:
        model = pickle.load(f)

    df_input = pd.DataFrame([employee])

    # make sure columns match training features
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    prob = model.predict_proba(df_input)[0][1]

    probability = round(float(prob), 3)

    # compare using the rounded value
    risk = "High" if probability >= 0.6 else "Low"

    return {
        "attrition_risk": risk,
        "probability": probability
    }