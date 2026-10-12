
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================
# 1. LOAD DATASETS
# =====================================

print("Loading OULAD datasets...")

student_info = pd.read_csv("dataset/studentInfo.csv")
assessments = pd.read_csv("dataset/assessments.csv")
student_assessment = pd.read_csv("dataset/studentAssessment.csv")
student_vle = pd.read_csv("dataset/studentVle.csv")

print("Student information:", student_info.shape)
print("Assessments:", assessments.shape)
print("Student assessment records:", student_assessment.shape)
print("Student VLE records:", student_vle.shape)


# =====================================
# 2. PREPARE ASSESSMENT FEATURES
# =====================================

print("\nPreparing assessment features...")

student_assessment["score"] = pd.to_numeric(
    student_assessment["score"],
    errors="coerce"
)

student_assessment = student_assessment.dropna(
    subset=["score"]
)

assessment_data = student_assessment.merge(
    assessments,
    on="id_assessment",
    how="left"
)

assessment_features = assessment_data.groupby(
    ["id_student", "code_module", "code_presentation"]
).agg(
    assessment_avg_score=("score", "mean"),
    assessment_count=("score", "count")
).reset_index()


# =====================================
# 3. PREPARE ONLINE ACTIVITY FEATURES
# =====================================

print("Preparing online activity features...")

student_vle["sum_click"] = pd.to_numeric(
    student_vle["sum_click"],
    errors="coerce"
).fillna(0)

student_vle["date"] = pd.to_numeric(
    student_vle["date"],
    errors="coerce"
)

activity_features = student_vle.groupby(
    ["id_student", "code_module", "code_presentation"]
).agg(
    total_clicks=("sum_click", "sum"),
    active_days=("date", "nunique")
).reset_index()


# =====================================
# 4. MERGE STUDENT DATA
# =====================================

print("Combining student records...")

data = student_info.merge(
    assessment_features,
    on=["id_student", "code_module", "code_presentation"],
    how="left"
)

data = data.merge(
    activity_features,
    on=["id_student", "code_module", "code_presentation"],
    how="left"
)


# =====================================
# 5. HANDLE MISSING VALUES
# =====================================

numeric_columns = [
    "assessment_avg_score",
    "assessment_count",
    "total_clicks",
    "active_days"
]

for column in numeric_columns:
    data[column] = data[column].fillna(0)

# Create required folders
os.makedirs("model", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# Save records for Student ID lookup
data.to_csv(
    "dataset/student_features.csv",
    index=False
)

print("Student records saved to dataset/student_features.csv")


# =====================================
# 6. SELECT MODEL FEATURES
# =====================================

features = [
    "code_module",
    "code_presentation",
    "gender",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "assessment_avg_score",
    "assessment_count",
    "total_clicks",
    "active_days"
]

target = "final_result"

X = data[features].copy()
y = data[target].copy()


# =====================================
# 7. CATEGORICAL AND NUMERIC FEATURES
# =====================================

categorical_features = [
    "code_module",
    "code_presentation",
    "gender",
    "disability"
]

numeric_features = [
    "num_of_prev_attempts",
    "studied_credits",
    "assessment_avg_score",
    "assessment_count",
    "total_clicks",
    "active_days"
]


# =====================================
# 8. PREPROCESS DATA
# =====================================

print("Preprocessing model features...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numeric_features
        )
    ]
)

X_processed = preprocessor.fit_transform(X)


# =====================================
# 9. SPLIT TRAINING AND TEST DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =====================================
# 10. TRAIN RANDOM FOREST
# =====================================

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)


# =====================================
# 11. EVALUATE MODEL
# =====================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

print("\n================================")
print("LEARNLIFT MODEL RESULTS")
print("================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =====================================
# 12. SAVE TRAINED MODEL
# =====================================

joblib.dump(
    model,
    "model/learnlift_model.pkl"
)

joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)

print("\nModel saved successfully.")


# =====================================
# 13. SAVE MODEL METRICS
# =====================================

metrics = {
    "accuracy": round(float(accuracy * 100), 2),
    "classification_report": report,
    "classes": [str(c) for c in model.classes_],
    "features": features
}

with open("model/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved successfully.")


# =====================================
# 14. RESULT DISTRIBUTION GRAPH
# =====================================

plt.figure(figsize=(8, 5))

data["final_result"].value_counts().plot(kind="bar")

plt.title("Student Final Result Distribution")
plt.xlabel("Final Result")
plt.ylabel("Number of Students")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "static/plots/result_distribution.png"
)

plt.close()


# =====================================
# 15. CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest")
plt.title("LearnLift Confusion Matrix")
plt.xlabel("Predicted Result")
plt.ylabel("Actual Result")

plt.xticks(
    range(len(model.classes_)),
    model.classes_,
    rotation=45
)

plt.yticks(
    range(len(model.classes_)),
    model.classes_
)

plt.colorbar()
plt.tight_layout()

plt.savefig(
    "static/plots/confusion_matrix.png"
)

plt.close()


# =====================================
# 16. MODEL ACCURACY GRAPH
# =====================================

plt.figure(figsize=(6, 5))

plt.bar(
    ["Random Forest"],
    [accuracy * 100]
)

plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("LearnLift Model Accuracy")

plt.tight_layout()

plt.savefig(
    "static/plots/model_accuracy.png"
)

plt.close()


# =====================================
# 17. TRAINING COMPLETE
# =====================================

print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================")

print("Model: model/learnlift_model.pkl")
print("Preprocessor: model/preprocessor.pkl")
print("Metrics: model/metrics.json")
print("Student lookup data: dataset/student_features.csv")
print("Graphs: static/plots/")
