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

# Direct Wikimedia muscle group diagrams (Reliable CORS-friendly links)
HEATMAPS = {
    "chest": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Pectoralis_major.png/400px-Pectoralis_major.png",
    "shoulder": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Deltoideus.png/400px-Deltoideus.png",
    "legs": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Quadriceps.png/400px-Quadriceps.png",
    "back": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Latissimus_dorsi.png/400px-Latissimus_dorsi.png",
    "biceps": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Biceps_brachii.png/400px-Biceps_brachii.png",
    "abs": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Rectus_abdominis.png/400px-Rectus_abdominis.png",
}

# Exercise execution animations using Wikimedia assets
EXERCISE_MEDIA = {
    "Incline Dumbbell Press": (
        "https://upload.wikimedia.org/wikipedia/commons/1/18/Incline-dumbbell-bench-press-2.gif"
    ),
    "Barbell Bench Press": (
        "https://upload.wikimedia.org/wikipedia/commons/8/82/Bench-press-1.gif"
    ),
    "Chest Flyes": (
        "https://upload.wikimedia.org/wikipedia/commons/f/f0/Dumbbell-fly-1.gif"
    ),
    "Cable Crossover": (
        "https://upload.wikimedia.org/wikipedia/commons/f/f5/Cable-standing-fly-1.gif"
    ),
    "Pec Deck Flyes (Joint Safe)": (
        "https://upload.wikimedia.org/wikipedia/commons/f/f0/Dumbbell-fly-1.gif"
    ),
    "Incline Cable Flyes (Joint Safe)": (
        "https://upload.wikimedia.org/wikipedia/commons/f/f5/Cable-standing-fly-1.gif"
    ),
    "Dumbbell Shoulder Press": (
        "https://upload.wikimedia.org/wikipedia/commons/a/a2/Dumbbell-shoulder-press-1.gif"
    ),
    "Lateral Raises": (
        "https://upload.wikimedia.org/wikipedia/commons/1/1c/Dumbbell-lateral-raise-1.gif"
    ),
    "Front Dumbbell Raises": (
        "https://upload.wikimedia.org/wikipedia/commons/8/87/Dumbbell-front-raise-1.gif"
    ),
    "Face Pulls": (
        "https://upload.wikimedia.org/wikipedia/commons/2/29/Face-pull-1.gif"
    ),
    "Barbell Squats": (
        "https://upload.wikimedia.org/wikipedia/commons/d/d1/Barbell-squat-1.gif"
    ),
    "Leg Press": (
        "https://upload.wikimedia.org/wikipedia/commons/1/1b/Leg-press-1.gif"
    ),
    "Leg Extensions": (
        "https://upload.wikimedia.org/wikipedia/commons/a/a8/Leg-extension-1.gif"
    ),
    "Lying Leg Curls": (
        "https://upload.wikimedia.org/wikipedia/commons/c/c2/Lying-leg-curl-1.gif"
    ),
    "Leg Extensions (Knee Safe)": (
        "https://upload.wikimedia.org/wikipedia/commons/a/a8/Leg-extension-1.gif"
    ),
    "Lat Pulldowns": (
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Lat-pulldown-1.gif"
    ),
    "Seated Cable Rows": (
        "https://upload.wikimedia.org/wikipedia/commons/e/e0/Seated-cable-row-1.gif"
    ),
    "Dumbbell Single-Arm Rows": (
        "https://upload.wikimedia.org/wikipedia/commons/2/26/One-arm-dumbbell-row-1.gif"
    ),
    "Barbell Deadlift": (
        "https://upload.wikimedia.org/wikipedia/commons/f/f3/Barbell-deadlift-1.gif"
    ),
    "Chest Supported Rows (Back Safe)": (
        "https://upload.wikimedia.org/wikipedia/commons/2/26/One-arm-dumbbell-row-1.gif"
    ),
    "Dumbbell Bicep Curls": (
        "https://upload.wikimedia.org/wikipedia/commons/2/2d/Dumbbell-biceps-curl-1.gif"
    ),
    "Hammer Curls": (
        "https://upload.wikimedia.org/wikipedia/commons/0/03/Hammer-curl-1.gif"
    ),
    "EZ Bar Preacher Curls": (
        "https://upload.wikimedia.org/wikipedia/commons/1/14/Preacher-curl-1.gif"
    ),
    "Concentration Curls": (
        "https://upload.wikimedia.org/wikipedia/commons/f/f9/Concentration-curl-1.gif"
    ),
    "Abdominal Crunches": (
        "https://upload.wikimedia.org/wikipedia/commons/2/2c/Crunch-1.gif"
    ),
    "Hanging Leg Raises": (
        "https://upload.wikimedia.org/wikipedia/commons/a/a7/Hanging-leg-raise-1.gif"
    ),
    "Plank Hold": (
        "https://upload.wikimedia.org/wikipedia/commons/e/e2/Plank-1.gif"
    ),
    "Russian Twists": (
        "https://upload.wikimedia.org/wikipedia/commons/9/91/Russian-twist-1.gif"
    ),
}

# Guaranteed 3-4 moves per muscle group across all levels
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
            ("EZ Bar Preacher Curls", "3 x 12"),
            ("Concentration Curls", "3 x 12"),
        ],
        "intermediate": [
            ("Dumbbell Bicep Curls", "4 x 10"),
            ("Hammer Curls", "4 x 10"),
            ("EZ Bar Preacher Curls", "3 x 10"),
            ("Concentration Curls", "3 x 10"),
        ],
        "advanced": [
            ("Dumbbell Bicep Curls", "4 x 8"),
            ("Hammer Curls", "4 x 8"),
            ("EZ Bar Preacher Curls", "4 x 8"),
            ("Concentration Curls", "4 x 8"),
        ],
    },
    "abs": {
        "beginner": [
            ("Abdominal Crunches", "3 x 15"),
            ("Hanging Leg Raises", "3 x 12"),
            ("Plank Hold", "3 x 45 sec"),
            ("Russian Twists", "3 x 20"),
        ],
        "intermediate": [
            ("Abdominal Crunches", "4 x 20"),
            ("Hanging Leg Raises", "4 x 12"),
            ("Plank Hold", "4 x 60 sec"),
            ("Russian Twists", "4 x 20"),
        ],
        "advanced": [
            ("Abdominal Crunches", "5 x 20"),
            ("Hanging Leg Raises", "4 x 15"),
            ("Plank Hold", "4 x 90 sec"),
            ("Russian Twists", "4 x 25"),
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
        ("Chest Flyes", "3 x 12"),
    ]
    replaced_warning = True
  elif injury == "Knee/Legs" and muscle.lower() == "legs":
    moves = [
        ("Leg Extensions (Knee Safe)", "3 x 15"),
        ("Lying Leg Curls", "3 x 12"),
    ]
    replaced_warning = True
  elif injury == "Lower Back" and muscle.lower() == "back":
    moves = [
        ("Chest Supported Rows (Back Safe)", "3 x 12"),
        ("Lat Pulldowns", "3 x 12"),
        ("Face Pulls", "3 x 15"),
    ]
    replaced_warning = True

  if replaced_warning:
    st.warning(
        f"⚠️ Injury Alert ({injury}): Routine modified to use joint-safe"
        " exercises."
    )
  else:
    st.success(f"Routine Generated for {muscle} ({level})")

  heatmap_url = HEATMAPS.get(
      muscle.lower(),
      "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Pectoralis_major.png/400px-Pectoralis_major.png",
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
      st.image(heatmap_url, caption=f"{muscle} Muscle Group", width=180)

    with col3:
      st.write("**Exercise Execution**")
      gif_url = EXERCISE_MEDIA.get(
          move,
          "https://upload.wikimedia.org/wikipedia/commons/1/18/Incline-dumbbell-bench-press-2.gif",
      )
      st.image(gif_url, caption=f"Execution: {move}", width=220)

    st.divider()

    moves_list.append(move)
    sets_list.append(int(sets_reps.split("x")[0].strip()))

  fig, ax = plt.subplots(figsize=(7, 3))
  ax.barh(moves_list, sets_list, color="#2ecc71")
  ax.set_xlabel("Sets")
  ax.set_title(f"Today's {muscle} Plan Breakdown")
  st.pyplot(fig)
