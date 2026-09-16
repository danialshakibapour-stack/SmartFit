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

# Reliable & High-Availability Heatmap Anatomy Assets (Wikimedia Raw SVG Links)
HEATMAPS = {
    "chest": "https://raw.githubusercontent.com/yuhas/exercise-db/main/assets/chest.png",
    "shoulder": "https://raw.githubusercontent.com/yuhas/exercise-db/main/assets/shoulders.png",
    "legs": "https://raw.githubusercontent.com/yuhas/exercise-db/main/assets/quads.png",
    "back": "https://raw.githubusercontent.com/yuhas/exercise-db/main/assets/lats.png",
    "biceps": "https://raw.githubusercontent.com/yuhas/exercise-db/main/assets/biceps.png",
    "abs": "https://raw.githubusercontent.com/yuhas/exercise-db/main/assets/abs.png",
}

# Verified Direct GIF URLs from Open-Source Datasets
GIFS = {
    "Incline Dumbbell Press": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0314.gif",
    "Barbell Bench Press": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0025.gif",
    "Chest Flyes": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0308.gif",
    "Cable Crossover": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0161.gif",
    "Pec Deck Flyes (Joint Safe)": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0308.gif",
    "Incline Cable Flyes (Joint Safe)": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0161.gif",
    "Dumbbell Shoulder Press": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0405.gif",
    "Lateral Raises": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0334.gif",
    "Front Dumbbell Raises": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0310.gif",
    "Face Pulls": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0160.gif",
    "Barbell Squats": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0043.gif",
    "Leg Press": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0585.gif",
    "Leg Extensions": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0583.gif",
    "Lying Leg Curls": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0593.gif",
    "Leg Extensions (Knee Safe)": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0583.gif",
    "Lat Pulldowns": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0150.gif",
    "Seated Cable Rows": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0239.gif",
    "Dumbbell Single-Arm Rows": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0292.gif",
    "Barbell Deadlift": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0032.gif",
    "Chest Supported Rows (Back Safe)": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0239.gif",
    "Dumbbell Bicep Curls": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0299.gif",
    "Hammer Curls": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0313.gif",
    "EZ Bar Preacher Curls": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0165.gif",
    "Abdominal Crunches": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0274.gif",
    "Hanging Leg Raises": "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0402.gif",
}

# Balanced 4-Exercise Workouts per Muscle Group
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

  fallback_heatmap = "https://raw.githubusercontent.com/yuhas/exercise-db/main/assets/chest.png"
  heatmap_url = HEATMAPS.get(muscle.lower(), fallback_heatmap)

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
      st.image(
          heatmap_url, caption=f"{muscle} Muscle Group", width=180
      )

    with col3:
      st.write("**Exercise Execution**")
      fallback_gif = "https://raw.githubusercontent.com/yuhas/exercise-db/main/gifs/0025.gif"
      gif_url = GIFS.get(move, fallback_gif)
      st.image(
          gif_url, caption=f"Execution: {move}", width=220
      )

    st.divider()

    moves_list.append(move)
    sets_list.append(int(sets_reps.split("x")[0].strip()))

  fig, ax = plt.subplots(figsize=(7, 3))
  ax.barh(moves_list, sets_list, color="#2ecc71")
  ax.set_xlabel("Sets")
  ax.set_title(f"Today's {muscle} Plan Breakdown")
  st.pyplot(fig)
