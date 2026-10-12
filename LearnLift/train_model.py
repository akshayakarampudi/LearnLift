import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt


# ==============================
# 1. Load datasets
# ==============================

student_info = pd.read_csv("dataset/studentInfo.csv")
assessments = pd.read_csv("dataset/assessments.csv")
student_assessment = pd.read_csv("dataset/studentAssessment.csv")
student_vle = pd.read_csv("dataset/studentVle.csv")


# ==============================
# 2. Assessment features
# ==============================

student_assessment["score"] = pd.to_numeric(
    student_assessment["score"],
    errors="coerce"
)

student_assessment = student_assessment.dropna(subset=["score"])

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


# ==============================
# 3. Online activity features
# ==============================

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


# ==============================
# 4. Merge all data
# ==============================

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


# ==============================
# 5. Fill missing values
# ==============================

data["assessment_avg_score"] = data[
    "assessment_avg_score"
].fillna(0)

data["assessment_count"] = data[
    "assessment_count"
].fillna(0)

data["total_clicks"] = data[
    "total_clicks"
].fillna(0)

data["active_days"] = data[
    "active_days"
].fillna(0)


# ==============================
# 6. Select features
# ==============================

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

X = data[features]
y = data["final_result"]


# ==============================
# 7. Categorical and numerical
# ==============================

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


# ==============================
# 8. Preprocessing
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


X_processed = preprocessor.fit_transform(X)


# ==============================
# 9. Train-test split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# 10. Train Random Forest
# ==============================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)


# ==============================
# 11. Prediction
# ==============================

y_pred = model.predict(X_test)


# ==============================
# 12. Accuracy
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("LearnLift Model Results")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==============================
# 13. Create model folder
# ==============================

os.makedirs("model", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)


# ==============================
# 14. Save model
# ==============================

joblib.dump(
    model,
    "model/learnlift_model.pkl"
)

joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)


# ==============================
# 15. Save metrics
# ==============================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

metrics = {
    "accuracy": round(accuracy * 100, 2),
    "classification_report": report
}

with open("model/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


# ==============================
# 16. Result distribution
# ==============================

plt.figure(figsize=(8, 5))

data["final_result"].value_counts().plot(
    kind="bar"
)

plt.title("Student Final Result Distribution")
plt.xlabel("Final Result")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig(
    "static/plots/result_distribution.png"
)

plt.close()


# ==============================
# 17. Confusion matrix
# ==============================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

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


# ==============================
# 18. Accuracy graph
# ==============================

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


print("\n==============================")
print("Training completed successfully!")
print("==============================")

print("\nModel saved:")
print("model/learnlift_model.pkl")

print("Preprocessor saved:")
print("model/preprocessor.pkl")

print("Metrics saved:")
print("model/metrics.json")

print("\nPlots saved inside:")
print("static/plots/")