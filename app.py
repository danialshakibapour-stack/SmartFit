import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="SmartFit AI", page_icon="🏋️‍♂️")

st.title("🏋️‍♂️ SmartFit: AI Workout Recommender")
st.write(
    "Enter your biometric metrics and workout preferences to get a"
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

EXERCISES = {
    "chest": {
        "beginner": [
            ("Push-ups", "3 x 10-12"),
            ("Dumbbell Bench Press", "3 x 10"),
        ],
        "intermediate": [
            ("Barbell Bench Press", "4 x 8-10"),
            ("Incline Dumbbell Press", "3 x 10"),
        ],
        "advanced": [
            ("Heavy Barbell Press", "5 x 5"),
            ("Weighted Dips", "4 x 8"),
        ],
    },
    "shoulder": {
        "beginner": [
            ("Dumbbell Shoulder Press", "3 x 12"),
            ("Lateral Raises", "3 x 15"),
        ],
        "intermediate": [
            ("Overhead Barbell Press", "4 x 8"),
            ("Arnold Press", "3 x 10"),
        ],
        "advanced": [
            ("Standing Military Press", "5 x 5"),
            ("Heavy Lateral Raises", "4 x 10"),
        ],
    },
    "legs": {
        "beginner": [("Goblet Squats", "3 x 12"), ("Leg Press", "3 x 12")],
        "intermediate": [
            ("Barbell Back Squats", "4 x 8"),
            ("Romanian Deadlifts", "3 x 10"),
        ],
        "advanced": [
            ("Heavy Barbell Squats", "5 x 5"),
            ("Bulgarian Split Squats", "4 x 8"),
        ],
    },
    "back": {
        "beginner": [
            ("Lat Pulldowns", "3 x 12"),
            ("Seated Cable Rows", "3 x 12"),
        ],
        "intermediate": [
            ("Pull-ups", "4 x 8"),
            ("Bent-Over Barbell Rows", "4 x 8"),
        ],
        "advanced": [("Deadlifts", "5 x 5"), ("Weighted Pull-ups", "4 x 6-8")],
    },
    "biceps": {
        "beginner": [
            ("Dumbbell Curls", "3 x 12"),
            ("Hammer Curls", "3 x 12"),
        ],
        "intermediate": [
            ("EZ-Bar Curls", "4 x 10"),
            ("Incline Dumbbell Curls", "3 x 10"),
        ],
        "advanced": [
            ("Preacher Curls", "4 x 8"),
            ("Concentration Curls", "3 x 12"),
        ],
    },
    "abs": {
        "beginner": [("Crunches", "3 x 15"), ("Plank", "3 x 30s")],
        "intermediate": [
            ("Hanging Leg Raises", "3 x 12"),
            ("Ab Wheel Rollouts", "3 x 10"),
        ],
        "advanced": [
            ("Weighted Leg Raises", "4 x 12"),
            ("Weighted Plank", "3 x 60s"),
        ],
    },
}

if st.sidebar.button("🚀 Generate Workout Routine"):
  moves = EXERCISES.get(muscle.lower(), {}).get(level.lower(), [])

  st.success(f"Routine Generated for {muscle} ({level})")

  moves_list, sets_list = [], []
  for idx, (move, sets_reps) in enumerate(moves, 1):
    st.write(f"### {idx}. {move}")
    st.info(f"Recommended Sets & Reps: {sets_reps}")
    moves_list.append(move)
    sets_list.append(int(sets_reps.split("x")[0].strip()))

  fig, ax = plt.subplots(figsize=(7, 3))
  ax.barh(moves_list, sets_list, color="#2ecc71")
  ax.set_xlabel("Sets")
  ax.set_title(f"Today's {muscle} Plan")
  st.pyplot(fig)
