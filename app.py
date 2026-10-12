import os
import json
import joblib
import pandas as pd

from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templates")

# Automatically reload updated HTML templates
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

MODEL_PATH = os.path.join(MODEL_DIR, "learnlift_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

FEATURES_PATH = os.path.join(DATASET_DIR, "student_features.csv")
CLUSTERS_PATH = os.path.join(MODEL_DIR, "student_clusters.csv")
CLUSTER_INFO_PATH = os.path.join(MODEL_DIR, "cluster_info.json")

# Features must remain in the same order used during model training
FEATURES = [
    "code_module",
    "code_presentation",
    "gender",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "assessment_avg_score",
    "assessment_count",
    "total_clicks",
    "active_days",
]


# ==========================================================
# LOAD MACHINE LEARNING FILES
# ==========================================================

model = None
preprocessor = None
metrics = {}

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print("WARNING: Model file not found:", MODEL_PATH)

    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Preprocessor loaded successfully.")
    else:
        print("WARNING: Preprocessor file not found:", PREPROCESSOR_PATH)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

except Exception:
    app.logger.exception("Error loading machine learning files")


# ==========================================================
# LOAD STUDENT DATA
# ==========================================================

students_df = pd.DataFrame()

try:
    if os.path.exists(FEATURES_PATH):
        students_df = pd.read_csv(FEATURES_PATH)

        students_df.columns = students_df.columns.str.strip()

        print("Student records loaded:", len(students_df))
        print("Student dataset columns:", students_df.columns.tolist())

    else:
        print("WARNING: Student features file not found:", FEATURES_PATH)

except Exception:
    app.logger.exception("Error loading student dataset")


# ==========================================================
# LOAD BEHAVIOUR CLUSTERING FILES
# ==========================================================

clusters_df = pd.DataFrame()
cluster_info = {}

try:
    if os.path.exists(CLUSTERS_PATH):
        clusters_df = pd.read_csv(CLUSTERS_PATH)
        clusters_df.columns = clusters_df.columns.str.strip()

        print("Student clusters loaded:", len(clusters_df))

    if os.path.exists(CLUSTER_INFO_PATH):
        with open(CLUSTER_INFO_PATH, "r", encoding="utf-8") as file:
            cluster_info = json.load(file)

except Exception:
    app.logger.exception("Error loading clustering files")


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================================
# DASHBOARD PAGE
# ==========================================================

@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        metrics=metrics,
        total_students=32593,
        pass_count=12361,
        distinction_count=3024,
        fail_count=7052,
        withdrawn_count=10156,
    )


# ==========================================================
# PREDICTION PAGE
# ==========================================================

@app.route("/prediction")
def prediction_page():

    modules = []
    presentations = []

    if not students_df.empty:

        if "code_module" in students_df.columns:
            modules = sorted(
                students_df["code_module"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        if "code_presentation" in students_df.columns:
            presentations = sorted(
                students_df["code_presentation"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

    return render_template(
        "prediction.html",
        modules=modules,
        presentations=presentations,
        students=students_df,
    )


# ==========================================================
# PREDICT STUDENT PERFORMANCE
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether the trained model and preprocessor are available
    if model is None or preprocessor is None:
        return (
            "Model files are missing. Please check "
            "model/learnlift_model.pkl and model/preprocessor.pkl.",
            500,
        )

    try:

        # Read form values from prediction.html
        student_id = request.form.get("id_student", "").strip()
        module = request.form.get("code_module", "").strip()
        presentation = request.form.get(
            "code_presentation", ""
        ).strip()

        # Validate submitted values
        if not student_id or not module or not presentation:
            return (
                "Please enter Student ID, Course Module and Presentation.",
                400,
            )

        # Check required dataset columns
        required_columns = [
            "id_student",
            "code_module",
            "code_presentation",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in students_df.columns
        ]

        if missing_columns:
            return (
                "Missing columns in student_features.csv: "
                + ", ".join(missing_columns),
                500,
            )

        # Convert IDs to numbers so values such as 11391
        # and 11391.0 can match
        student_ids = pd.to_numeric(
            students_df["id_student"],
            errors="coerce",
        )

        entered_id = pd.to_numeric(
            student_id,
            errors="coerce",
        )

        # Find the student using ID, module and presentation
        matching = students_df[
            (student_ids == entered_id)
            & (
                students_df["code_module"].astype(str).str.strip()
                == module
            )
            & (
                students_df["code_presentation"].astype(str).str.strip()
                == presentation
            )
        ]

        # Stop if the record does not exist
        if matching.empty:
            return (
                "Student record not found. Please select a valid "
                "Student ID, Course Module and Presentation.",
                404,
            )

        # Select the matching student's information
        student = matching.iloc[[0]].copy()

        # Verify that all model features exist
        missing_features = [
            feature
            for feature in FEATURES
            if feature not in student.columns
        ]

        if missing_features:
            return (
                "Missing model features in student_features.csv: "
                + ", ".join(missing_features),
                500,
            )

        # Prepare features in the correct training order
        X = student[FEATURES]

        # Apply the saved preprocessing pipeline
        X_processed = preprocessor.transform(X)

        # Predict the academic outcome
        prediction = model.predict(X_processed)[0]

        # Initialize prediction results
        confidence = None
        probabilities = {}

        # Calculate probability for each outcome
        if hasattr(model, "predict_proba"):

            class_probabilities = model.predict_proba(X_processed)[0]

            # Convert probabilities to percentages
            probabilities = {
                str(label): round(float(probability) * 100, 1)
                for label, probability in zip(
                    model.classes_,
                    class_probabilities,
                )
            }

            # Confidence is the probability assigned to the predicted class
            predicted_label = str(prediction)

            confidence = probabilities.get(predicted_label)

            if confidence is None:
                confidence = round(
                    float(max(class_probabilities)) * 100,
                    1,
                )

        # Simple risk indicator based on predicted outcome.
        # This is not a separately trained dropout-risk model.
        if str(prediction).strip().lower() == "withdrawn":
            dropout_risk = "High"
        else:
            dropout_risk = "Lower"

        # Display the result page
        return render_template(
            "result.html",
            prediction=str(prediction),
            confidence=confidence,
            dropout_risk=dropout_risk,
            probabilities=probabilities,
            student_id=student_id,
            module=module,
            presentation=presentation,
        )

    except Exception as error:

        app.logger.exception("Prediction failed")

        return (
            "Prediction error: " + str(error),
            500,
        )


# ==========================================================
# BEHAVIOUR HELPER FUNCTIONS
# ==========================================================

def get_cluster_name(cluster_id):
    """Get a descriptive cluster name from cluster_info.json."""

    if isinstance(cluster_info, dict):

        info = cluster_info.get(str(cluster_id), {})

        if isinstance(info, dict):
            for key in ("name", "label", "cluster_name"):
                if key in info:
                    return str(info[key])

        if isinstance(info, str):
            return info

        cluster_list = cluster_info.get("clusters", [])

        if isinstance(cluster_list, list):

            for item in cluster_list:

                if isinstance(item, dict):

                    item_id = item.get(
                        "cluster",
                        item.get("cluster_id"),
                    )

                    if str(item_id) == str(cluster_id):
                        return str(
                            item.get(
                                "name",
                                item.get(
                                    "label",
                                    f"Cluster {cluster_id}",
                                ),
                            )
                        )

    # Fallback labels. Confirm these match your clustering output.
    fallback = {
        "0": "Highly Active Learner",
        "1": "Low Activity Learner",
        "2": "Moderate Activity Learner",
    }

    return fallback.get(
        str(cluster_id),
        f"Cluster {cluster_id}",
    )


def classify_cluster_name(name):
    """Convert a cluster description into a display group."""

    label = str(name).lower()

    if "high" in label:
        return "high"

    if "moderate" in label or "medium" in label:
        return "moderate"

    if "low" in label:
        return "low"

    return "unknown"


# ==========================================================
# BEHAVIOUR PAGE
# ==========================================================

@app.route("/behaviour")
def behaviour():

    high_count = 0
    moderate_count = 0
    low_count = 0
    cluster_error = None

    if clusters_df.empty:

        cluster_error = (
            "Clustering results were not found. "
            "Check model/student_clusters.csv."
        )

    else:

        cluster_column = next(
            (
                column
                for column in clusters_df.columns
                if column.lower() in (
                    "cluster",
                    "cluster_id",
                    "cluster_label",
                    "kmeans_cluster",
                )
            ),
            None,
        )

        if cluster_column is None:

            cluster_column = next(
                (
                    column
                    for column in clusters_df.columns
                    if "cluster" in column.lower()
                ),
                None,
            )

        if cluster_column is None:

            cluster_error = (
                "Could not find a cluster column in "
                "student_clusters.csv. Please check its column names."
            )

        else:

            for value in clusters_df[cluster_column].dropna():

                name = get_cluster_name(value)
                group = classify_cluster_name(name)

                if group == "unknown":
                    group = classify_cluster_name(value)

                if group == "high":
                    high_count += 1

                elif group == "moderate":
                    moderate_count += 1

                elif group == "low":
                    low_count += 1

    total_students = high_count + moderate_count + low_count

    return render_template(
        "behaviour.html",
        total_students=total_students,
        high_count=high_count,
        moderate_count=moderate_count,
        low_count=low_count,
        cluster_error=cluster_error,
    )


# ==========================================================
# MODEL PERFORMANCE PAGE
# ==========================================================

@app.route("/model")
def model_page():
    return render_template(
        "model.html",
        metrics=metrics,
    )


# ==========================================================
# ABOUT PAGE
# ==========================================================

@app.route("/about")
def about():
    return render_template("about.html")


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    print("LearnLift project folder:", BASE_DIR)
    print("Templates folder:", os.path.join(BASE_DIR, "templates"))
    print("Plots folder:", PLOTS_DIR)

    app.run(debug=True)