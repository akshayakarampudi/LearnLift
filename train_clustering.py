import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ==========================================
# FILE PATHS
# ==========================================

DATASET_PATH = "dataset"
MODEL_PATH = "model"


# ==========================================
# CREATE MODEL FOLDER
# ==========================================

os.makedirs(MODEL_PATH, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

print("=" * 60)
print("LearnLift - Student Behaviour Clustering")
print("=" * 60)

print("\nLoading datasets...")


student_info = pd.read_csv(
    os.path.join(
        DATASET_PATH,
        "studentInfo.csv"
    )
)

student_assessment = pd.read_csv(
    os.path.join(
        DATASET_PATH,
        "studentAssessment.csv"
    )
)

student_vle = pd.read_csv(
    os.path.join(
        DATASET_PATH,
        "studentVle.csv"
    )
)


print("Student information:", student_info.shape)
print("Student assessment:", student_assessment.shape)
print("Student VLE:", student_vle.shape)


# ==========================================
# ASSESSMENT FEATURES
# ==========================================

print("\nCreating assessment features...")


student_assessment["score"] = pd.to_numeric(
    student_assessment["score"],
    errors="coerce"
)


student_assessment = student_assessment.dropna(
    subset=["score"]
)


assessment_features = (
    student_assessment
    .groupby("id_student")
    .agg(
        assessment_avg_score=(
            "score",
            "mean"
        ),

        assessment_count=(
            "score",
            "count"
        )
    )
    .reset_index()
)


# ==========================================
# ONLINE ACTIVITY FEATURES
# ==========================================

print("Creating online learning activity features...")


student_vle["sum_click"] = pd.to_numeric(
    student_vle["sum_click"],
    errors="coerce"
).fillna(0)


student_vle["date"] = pd.to_numeric(
    student_vle["date"],
    errors="coerce"
)


activity_features = (
    student_vle
    .groupby("id_student")
    .agg(
        total_clicks=(
            "sum_click",
            "sum"
        ),

        active_days=(
            "date",
            "nunique"
        )
    )
    .reset_index()
)


# ==========================================
# STUDENT LIST
# ==========================================

students = student_info[
    ["id_student"]
].drop_duplicates()


# ==========================================
# MERGE FEATURES
# ==========================================

print("\nCombining student features...")


student_features = students.merge(
    assessment_features,
    on="id_student",
    how="left"
)


student_features = student_features.merge(
    activity_features,
    on="id_student",
    how="left"
)


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

feature_columns = [
    "assessment_avg_score",
    "assessment_count",
    "total_clicks",
    "active_days"
]


student_features[
    feature_columns
] = student_features[
    feature_columns
].fillna(0)


# ==========================================
# REMOVE EXTREME VALUES
# ==========================================

# Log transformation helps reduce the effect
# of very large click counts.

student_features["total_clicks"] = np.log1p(
    student_features["total_clicks"]
)


# ==========================================
# PREPARE DATA FOR CLUSTERING
# ==========================================

X = student_features[
    feature_columns
]


# ==========================================
# SCALE FEATURES
# ==========================================

print("\nScaling features...")


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# K-MEANS CLUSTERING
# ==========================================

print("\nTraining K-Means clustering model...")


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


student_features["cluster"] = kmeans.fit_predict(
    X_scaled
)


# ==========================================
# ANALYZE CLUSTERS
# ==========================================

cluster_summary = (
    student_features
    .groupby("cluster")
    [feature_columns]
    .mean()
)


print("\nCluster Summary:")
print(cluster_summary)


# ==========================================
# IDENTIFY CLUSTER TYPES
# ==========================================

# Calculate an overall activity score
# to give meaningful names to clusters.

cluster_activity = (
    cluster_summary["assessment_avg_score"]
    + cluster_summary["assessment_count"]
    + cluster_summary["total_clicks"]
    + cluster_summary["active_days"]
)


sorted_clusters = cluster_activity.sort_values().index.tolist()


cluster_names = {}


if len(sorted_clusters) == 3:

    cluster_names[
        sorted_clusters[0]
    ] = "Low Activity Learner"

    cluster_names[
        sorted_clusters[1]
    ] = "Moderate Activity Learner"

    cluster_names[
        sorted_clusters[2]
    ] = "Highly Active Learner"


student_features["cluster_name"] = (
    student_features["cluster"]
    .map(cluster_names)
)


# ==========================================
# PRINT RESULTS
# ==========================================

print("\nStudent Behaviour Groups:")

print(
    student_features[
        [
            "cluster",
            "cluster_name"
        ]
    ]
    .value_counts()
    .sort_index()
)


# ==========================================
# SAVE MODELS
# ==========================================

joblib.dump(
    kmeans,
    os.path.join(
        MODEL_PATH,
        "clustering_model.pkl"
    )
)


joblib.dump(
    scaler,
    os.path.join(
        MODEL_PATH,
        "clustering_scaler.pkl"
    )
)


# ==========================================
# SAVE CLUSTER INFORMATION
# ==========================================

cluster_counts = (
    student_features[
        "cluster_name"
    ]
    .value_counts()
    .to_dict()
)


cluster_info = {

    "clusters": cluster_names,

    "cluster_counts": cluster_counts,

    "features": feature_columns
}


with open(
    os.path.join(
        MODEL_PATH,
        "cluster_info.json"
    ),
    "w"
) as file:

    json.dump(
        cluster_info,
        file,
        indent=4
    )


# ==========================================
# SAVE STUDENT CLUSTER DATA
# ==========================================

student_features.to_csv(
    os.path.join(
        MODEL_PATH,
        "student_clusters.csv"
    ),
    index=False
)


# ==========================================
# FINISHED
# ==========================================

print("\n" + "=" * 60)

print(
    "Student behaviour clustering completed successfully!"
)

print("=" * 60)

print("\nCreated files:")

print(
    "model/clustering_model.pkl"
)

print(
    "model/clustering_scaler.pkl"
)

print(
    "model/cluster_info.json"
)

print(
    "model/student_clusters.csv"
)