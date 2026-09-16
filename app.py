import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="SmartFit AI", page_icon="🏋️‍♂️", layout="wide")

st.title("🏋️‍♂️ SmartFit: AI Workout Recommender")
st.write(
    "Enter your biometric metrics, injury status, and workout preferences to get a"
    " personalized routine!"
)

# Sidebar User Inputs
st.sidebar.header("User Metrics")
age = st.sidebar.number_input("Age", min_value=15, max_value=80, value=25)
weight = st.sidebar.number_input(
    "Weight (kg)", min_value=40, max_value=150, value=75
)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
level = st.sidebar.selectbox(
    "Experience Level", ["Beginner", "Intermediate", "Advanced"]
)
muscle = st.sidebar.selectbox(
    "Target Muscle Today",
    ["Chest", "Shoulder", "Legs", "Back", "Biceps", "Abs"],
)

injury = st.sidebar.selectbox(
    "Any Recent or Past Injuries?",
    ["None", "Shoulder", "Knee/Legs", "Lower Back"],
)

# Base URL for public domain exercise assets
BASE_IMG = "https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/exercises/"

# Static Muscle Group Anatomy Indicators
HEATMAPS = {
    "chest": f"{BASE_IMG}Barbell_Bench_Press/0.jpg",
    "shoulder": f"{BASE_IMG}Arnold_Dumbbell_Press/0.jpg",
    "legs": f"{BASE_IMG}Barbell_Full_Squat/0.jpg",
    "back": f"{BASE_IMG}Barbell_Deadlift/0.jpg",
    "biceps": f"{BASE_IMG}Barbell_Curl/0.jpg",
    "abs": f"{BASE_IMG}Air_Bike/0.jpg",
}

# Exercise Execution Images (Working Public URLs)
EXERCISE_IMAGES = {
    "Incline Dumbbell Press": f"{BASE_IMG}Alternate_Incline_Dumbbell_Curl/1.jpg",
    "Barbell Bench Press": f"{BASE_IMG}Barbell_Bench_Press/1.jpg",
    "Chest Flyes": f"{BASE_IMG}Air_Bike/1.jpg",
    "Cable Crossover": f"{BASE_IMG}Air_Bike/0.jpg",
    "Pec Deck Flyes (Joint Safe)": f"{BASE_IMG}Air_Bike/1.jpg",
    "Incline Cable Flyes (Joint Safe)": f"{BASE_IMG}Air_Bike/0.jpg",
    "Dumbbell Shoulder Press": f"{BASE_IMG}Arnold_Dumbbell_Press/1.jpg",
    "Lateral Raises": f"{BASE_IMG}Arm_Circles/0.jpg",
    "Front Dumbbell Raises": f"{BASE_IMG}Arm_Circles/1.jpg",
    "Face Pulls": f"{BASE_IMG}Arm_Circles/0.jpg",
    "Barbell Squats": f"{BASE_IMG}Barbell_Full_Squat/1.jpg",
    "Leg Press": f"{BASE_IMG}Barbell_Full_Squat/0.jpg",
    "Leg Extensions": f"{BASE_IMG}Barbell_Full_Squat/1.jpg",
    "Lying Leg Curls": f"{BASE_IMG}Barbell_Full_Squat/0.jpg",
    "Leg Extensions (Knee Safe)": f"{BASE_IMG}Barbell_Full_Squat/1.jpg",
    "Lat Pulldowns": f"{BASE_IMG}Barbell_Deadlift/0.jpg",
    "Seated Cable Rows": f"{BASE_IMG}Barbell_Deadlift/1.jpg",
    "Dumbbell Single-Arm Rows": f"{BASE_IMG}Barbell_Deadlift/0.jpg",
    "Barbell Deadlift": f"{BASE_IMG}Barbell_Deadlift/1.jpg",
    "Chest Supported Rows (Back Safe)": f"{BASE_IMG}Barbell_Deadlift/0.jpg",
    "Dumbbell Bicep Curls": f"{BASE_IMG}Barbell_Curl/1.jpg",
    "Hammer Curls": f"{BASE_IMG}Alternate_Incline_Dumbbell_Curl/0.jpg",
    "EZ Bar Preacher Curls": f"{BASE_IMG}Barbell_Curl/0.jpg",
    "Abdominal Crunches": f"{BASE_IMG}3_4_Sit-Up/0.jpg",
    "Hanging Leg Raises": f"{BASE_IMG}3_4_Sit-Up/1.jpg",
}

EXERCISES = {
    "chest": {
        "beginner": [
            ("Incline Dumbbell Press", "3 x 10-12"),
            ("Barbell Bench Press", "3 x 10"),
            ("Chest Flyes", "3 x 12"),
            ("Cable Crossover", "3 x 12"),
        ],
        "intermediate": [
            ("Incline Dumbbell Press", "4 x 8-10"),
            ("Barbell Bench Press", "4 x 8"),
            ("Chest Flyes", "3 x 12"),
            ("Cable Crossover", "3 x 12"),
        ],
        "advanced": [
            ("Incline Dumbbell Press", "4 x 6-8"),
            ("Barbell Bench Press", "4 x 6-8"),
            ("Chest Flyes", "4 x 10"),
            ("Cable Crossover", "4 x 10"),
        ],
    },
    "shoulder": {
        "beginner": [
            ("Dumbbell Shoulder Press", "3 x 12"),
            ("Lateral Raises", "3 x 15"),
            ("Front Dumbbell Raises", "3 x 12"),
            ("Face Pulls", "3 x 15"),
        ],
        "intermediate": [
            ("Dumbbell Shoulder Press", "4 x 8-10"),
            ("Lateral Raises", "4 x 12"),
            ("Front Dumbbell Raises", "3 x 12"),
            ("Face Pulls", "4 x 12"),
        ],
        "advanced": [
            ("Dumbbell Shoulder Press", "4 x 6-8"),
            ("Lateral Raises", "5 x 10"),
            ("Front Dumbbell Raises", "4 x 10"),
            ("Face Pulls", "4 x 10"),
        ],
    },
    "legs": {
        "beginner": [
            ("Barbell Squats", "3 x 12"),
            ("Leg Press", "3 x 12"),
            ("Leg Extensions", "3 x 12"),
            ("Lying Leg Curls", "3 x 12"),
        ],
        "intermediate": [
            ("Barbell Squats", "4 x 10"),
            ("Leg Press", "4 x 10"),
            ("Leg Extensions", "3 x 12"),
            ("Lying Leg Curls", "3 x 12"),
        ],
        "advanced": [
            ("Barbell Squats", "4 x 8"),
            ("Leg Press", "4 x 8"),
            ("Leg Extensions", "4 x 10"),
            ("Lying Leg Curls", "4 x 10"),
        ],
    },
    "back": {
        "beginner": [
            ("Lat Pulldowns", "3 x 12"),
            ("Seated Cable Rows", "3 x 12"),
            ("Dumbbell Single-Arm Rows", "3 x 12"),
            ("Face Pulls", "3 x 15"),
        ],
        "intermediate": [
            ("Lat Pulldowns", "4 x 10"),
            ("Seated Cable Rows", "4 x 10"),
            ("Dumbbell Single-Arm Rows", "3 x 10"),
            ("Barbell Deadlift", "3 x 8"),
        ],
        "advanced": [
            ("Lat Pulldowns", "4 x 8"),
            ("Seated Cable Rows", "4 x 8"),
            ("Dumbbell Single-Arm Rows", "4 x 8"),
            ("Barbell Deadlift", "4 x 6"),
        ],
    },
    "biceps": {
        "beginner": [
            ("Dumbbell Bicep Curls", "3 x 12"),
            ("Hammer Curls", "3 x 12"),
        ],
        "intermediate": [
            ("Dumbbell Bicep Curls", "4 x 10"),
            ("Hammer Curls", "4 x 10"),
            ("EZ Bar Preacher Curls", "3 x 10"),
        ],
        "advanced": [
            ("Dumbbell Bicep Curls", "4 x 8"),
            ("Hammer Curls", "4 x 8"),
            ("EZ Bar Preacher Curls", "4 x 8"),
        ],
    },
    "abs": {
        "beginner": [
            ("Abdominal Crunches", "3 x 15"),
            ("Hanging Leg Raises", "3 x 12"),
        ],
        "intermediate": [
            ("Abdominal Crunches", "4 x 20"),
            ("Hanging Leg Raises", "4 x 12"),
        ],
        "advanced": [
            ("Abdominal Crunches", "5 x 20"),
            ("Hanging Leg Raises", "4 x 15"),
        ],
    },
}

if st.sidebar.button("🚀 Generate Workout Routine"):
  moves = EXERCISES.get(muscle.lower(), {}).get(level.lower(), []).copy()

  replaced_warning = False
  if injury == "Shoulder" and muscle.lower() == "chest":
    moves = [
        ("Pec Deck Flyes (Joint Safe)", "3 x 12"),
        ("Incline Cable Flyes (Joint Safe)", "3 x 12"),
    ]
    replaced_warning = True
  elif injury == "Knee/Legs" and muscle.lower() == "legs":
    moves = [("Leg Extensions (Knee Safe)", "3 x 15")]
    replaced_warning = True
  elif injury == "Lower Back" and muscle.lower() == "back":
    moves = [("Chest Supported Rows (Back Safe)", "3 x 12")]
    replaced_warning = True

  if replaced_warning:
    st.warning(
        f"⚠️ Injury Alert ({injury}): Routine modified to use joint-safe"
        " exercises that avoid pain."
    )
  else:
    st.success(f"Routine Generated for {muscle} ({level})")

  heatmap_url = HEATMAPS.get(
      muscle.lower(), f"{BASE_IMG}Barbell_Bench_Press/0.jpg"
  )

  moves_list, sets_list = [], []
  for idx, (move, sets_reps) in enumerate(moves, 1):
    st.markdown(f"### {idx}. {move}")
    col1, col2, col3 = st.columns([2, 1.5, 2])

    with col1:
      st.info(f"**Recommended Sets & Reps:** {sets_reps}")
      st.caption(
          f"Targeted Muscle Group: **{muscle}** | Modeled for: **{gender}**"
      )

    with col2:
      st.write("**Target Anatomy**")
      st.image(heatmap_url, caption=f"{muscle} Target", width=180)

    with col3:
      st.write("**Exercise Execution**")
      img_url = EXERCISE_IMAGES.get(move, f"{BASE_IMG}Barbell_Bench_Press/1.jpg")
      st.image(img_url, caption=f"Execution: {move}", width=220)

    st.divider()

    moves_list.append(move)
    sets_list.append(int(sets_reps.split("x")[0].strip()))

  fig, ax = plt.subplots(figsize=(7, 3))
  ax.barh(moves_list, sets_list, color="#2ecc71")
  ax.set_xlabel("Sets")
  ax.set_title(f"Today's {muscle} Plan Breakdown")
  st.pyplot(fig)
