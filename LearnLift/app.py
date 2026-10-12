from flask import Flask, render_template, request
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("model/learnlift_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

# ============================================================
# LOAD MODEL METRICS
# ============================================================

metrics = {}

if os.path.exists("model/metrics.json"):
    with open("model/metrics.json", "r") as file:
        metrics = json.load(file)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html", metrics=metrics)


# ============================================================
# PREDICTION PAGE
# ============================================================

@app.route("/prediction")
def prediction_page():
    return render_template("prediction.html")


# ============================================================
# PREDICT STUDENT PERFORMANCE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # GET DATA FROM FORM
        # ----------------------------------------------------

        data = {
            "code_module": request.form.get("code_module", "AAA"),
            "code_presentation": request.form.get(
                "code_presentation", "2013J"
            ),

            "gender": request.form.get("gender", "Unknown"),

            # These 4 fields were removed from your visible form.
            # We give them a safe unknown value so the current
            # saved model can still receive the required columns.
            "region": request.form.get("region", "Unknown"),
            "highest_education": request.form.get(
                "highest_education", "Unknown"
            ),
            "imd_band": request.form.get(
                "imd_band", "Unknown"
            ),
            "age_band": request.form.get(
                "age_band", "Unknown"
            ),

            "num_of_prev_attempts": int(
                request.form.get("num_of_prev_attempts", 0)
            ),

            "studied_credits": int(
                request.form.get("studied_credits", 0)
            ),

            "disability": request.form.get(
                "disability", "N"
            ),

            "assessment_avg_score": float(
                request.form.get("assessment_avg_score", 0)
            ),

            "assessment_count": int(
                request.form.get("assessment_count", 0)
            ),

            "total_clicks": int(
                request.form.get("total_clicks", 0)
            ),

            "active_days": int(
                request.form.get("active_days", 0)
            )
        }

        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame([data])

        # ----------------------------------------------------
        # PREPROCESS INPUT
        # ----------------------------------------------------

        input_processed = preprocessor.transform(input_data)

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_processed)[0]

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        probabilities = model.predict_proba(input_processed)[0]

        classes = model.classes_

        probability_data = {}

        for class_name, probability in zip(classes, probabilities):
            probability_data[str(class_name)] = round(
                float(probability) * 100, 2
            )

        # ----------------------------------------------------
        # MAKE SURE ALL FOUR RESULT CATEGORIES EXIST
        # ----------------------------------------------------

        distinction_probability = probability_data.get(
            "Distinction", 0
        )

        fail_probability = probability_data.get(
            "Fail", 0
        )

        pass_probability = probability_data.get(
            "Pass", 0
        )

        withdrawn_probability = probability_data.get(
            "Withdrawn", 0
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = max(probabilities) * 100

        # ----------------------------------------------------
        # DROPOUT RISK
        # ----------------------------------------------------

        if prediction == "Withdrawn":
            dropout_risk = "High"

        elif prediction == "Fail":
            dropout_risk = "Medium"

        else:
            dropout_risk = "Low"

        # ----------------------------------------------------
        # SEND DATA TO RESULT PAGE
        # ----------------------------------------------------

        return render_template(
            "result.html",

            prediction=prediction,

            confidence=round(confidence, 2),

            dropout_risk=dropout_risk,

            distinction_probability=distinction_probability,

            fail_probability=fail_probability,

            pass_probability=pass_probability,

            withdrawn_probability=withdrawn_probability
        )

    except Exception as e:

        return render_template(
            "prediction.html",
            error=str(e)
        )


# ============================================================
# MODEL PAGE
# ============================================================

@app.route("/model")
def model_page():

    return render_template(
        "model.html",
        metrics=metrics
    )


# ============================================================
# ABOUT PAGE
# ============================================================

@app.route("/about")
def about():

    return render_template("about.html")


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)