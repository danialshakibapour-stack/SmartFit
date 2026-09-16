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

# 1. Injury Filter Input
injury = st.sidebar.selectbox(
    "Any Recent or Past Injuries?",
    ["None", "Shoulder", "Knee/Legs", "Lower Back"],
)

# Visual Asset URLs organized by Gender and Muscle/Exercise
MALE_HEATMAPS = {
    "chest": "https://raw.githubusercontent.com/pubf/muscle-assets/main/male_chest.png",
    "shoulder": "https://raw.githubusercontent.com/pubf/muscle-assets/main/male_shoulder.png",
    "legs": "https://raw.githubusercontent.com/pubf/muscle-assets/main/male_legs.png",
    "back": "https://raw.githubusercontent.com/pubf/muscle-assets/main/male_back.png",
    "biceps": "https://raw.githubusercontent.com/pubf/muscle-assets/main/male_biceps.png",
    "abs": "https://raw.githubusercontent.com/pubf/muscle-assets/main/male_abs.png",
}

FEMALE_HEATMAPS = {
    "chest": "https://raw.githubusercontent.com/pubf/muscle-assets/main/female_chest.png",
    "shoulder": "https://raw.githubusercontent.com/pubf/muscle-assets/main/female_shoulder.png",
    "legs": "https://raw.githubusercontent.com/pubf/muscle-assets/main/female_legs.png",
    "back": "https://raw.githubusercontent.com/pubf/muscle-assets/main/female_back.png",
    "biceps": "https://raw.githubusercontent.com/pubf/muscle-assets/main/female_biceps.png",
    "abs": "https://raw.githubusercontent.com/pubf/muscle-assets/main/female_abs.png",
}

GIFS = {
    "Male": {
        "Push-ups": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdW5pMXkycnRocjhkNDJ2ZmZzbWFmYmIwbzNneDFyeGgzaDFmdDFxeSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/k3xB4Pxf5qA95lVzO8/giphy.gif",
        "Dumbbell Bench Press": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3YxYnlyYmx1MnRvdmlsaWczNDc5OWZ3aGV0OXUwa2J5czNzeWtlZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKRBB3E7yqG3Lq0/giphy.gif",
        "Pec Deck Flyes (Joint Safe)": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjRxMXE4ZDV3MWt5YnAxeW1iZzZwbWZwbmd2ZnhuOG9vYzR5cjB5NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26E3ZfMhSAt7R7A36/giphy.gif",
        "Dumbbell Shoulder Press": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOWQyb3VpMndicnpodXBhMnI3NWpxbzVvZmpseHpwMnM1NDcxdDFjcyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKSjRrfIPjeiVyM/giphy.gif",
        "Lateral Raises": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdW9uZGs2ZXFlNm9oYmljcHRiMnIybXh1NDBicXltNWpvemlreTl2OCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0HlCqV38B42vN5QY/giphy.gif",
        "Goblet Squats": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMml1MWo4eTRtbDFkODRhNnJyZXUzdmprcGNqZzBveGdtb3U0OHQxOCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKU8RvQuomF5Rvy/giphy.gif",
        "Leg Extensions (Knee Safe)": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDh1NDIxbmsyc296dWZkMmlyeXR2dTBwbHV5NmhsaXdxMnRvdXk3ayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKP4S5sN7L3f1uE/giphy.gif",
        "Lat Pulldowns": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYnJ2dnFwZzNsb3FiYWFqMGFtbXZ3MmIwbWR2ZXRqMWY4MGswMW4zeCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKDkDbIDJ1zO69q/giphy.gif",
        "Chest Supported Rows (Back Safe)": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbjRndDRsdXpxa3VwNXlsMDRtdDF5bzBzMHByNWUzaGs2OG8yMnE5ciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKOn40D8l1c3vMc/giphy.gif",
        "Dumbbell Curls": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNndxYnhrZWJ5ZjNsbWtoazB3YXVqZnR5Ynpxb2c5eWFnbWZkbmcyYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0HlSno0N42F3zZ5S/giphy.gif",
        "Crunches": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDdwZHFlNnYydmpxenYyb3ZldG03azJkOXN2MnRhMjVqdHZkcmZhbCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKT046D5eFpZp5C/giphy.gif",
    },
    "Female": {
        "Push-ups": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdW5pMXkycnRocjhkNDJ2ZmZzbWFmYmIwbzNneDFyeGgzaDFmdDFxeSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/k3xB4Pxf5qA95lVzO8/giphy.gif",
        "Dumbbell Bench Press": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3YxYnlyYmx1MnRvdmlsaWczNDc5OWZ3aGV0OXUwa2J5czNzeWtlZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKRBB3E7yqG3Lq0/giphy.gif",
        "Pec Deck Flyes (Joint Safe)": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjRxMXE4ZDV3MWt5YnAxeW1iZzZwbWZwbmd2ZnhuOG9vYzR5cjB5NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26E3ZfMhSAt7R7A36/giphy.gif",
        "Dumbbell Shoulder Press": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOWQyb3VpMndicnpodXBhMnI3NWpxbzVvZmpseHpwMnM1NDcxdDFjcyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKSjRrfIPjeiVyM/giphy.gif",
        "Lateral Raises": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdW9uZGs2ZXFlNm9oYmljcHRiMnIybXh1NDBicXltNWpvemlreTl2OCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0HlCqV38B42vN5QY/giphy.gif",
        "Goblet Squats": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMml1MWo4eTRtbDFkODRhNnJyZXUzdmprcGNqZzBveGdtb3U0OHQxOCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKU8RvQuomF5Rvy/giphy.gif",
        "Leg Extensions (Knee Safe)": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDh1NDIxbmsyc296dWZkMmlyeXR2dTBwbHV5NmhsaXdxMnRvdXk3ayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKP4S5sN7L3f1uE/giphy.gif",
        "Lat Pulldowns": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYnJ2dnFwZzNsb3FiYWFqMGFtbXZ3MmIwbWR2ZXRqMWY4MGswMW4zeCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKDkDbIDJ1zO69q/giphy.gif",
        "Chest Supported Rows (Back Safe)": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbjRndDRsdXpxa3VwNXlsMDRtdDF5bzBzMHByNWUzaGs2OG8yMnE5ciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKOn40D8l1c3vMc/giphy.gif",
        "Dumbbell Curls": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNndxYnhrZWJ5ZjNsbWtoazB3YXVqZnR5Ynpxb2c5eWFnbWZkbmcyYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0HlSno0N42F3zZ5S/giphy.gif",
        "Crunches": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDdwZHFlNnYydmpxenYyb3ZldG03azJkOXN2MnRhMjVqdHZkcmZhbCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKT046D5eFpZp5C/giphy.gif",
    },
}

# Base Exercise Dictionary
EXERCISES = {
    "chest": {
        "beginner": [
            ("Push-ups", "3 x 10-12"),
            ("Dumbbell Bench Press", "3 x 10"),
        ],
        "intermediate": [
            ("Dumbbell Bench Press", "4 x 8-10"),
            ("Push-ups", "3 x 15"),
        ],
        "advanced": [
            ("Dumbbell Bench Press", "5 x 5"),
            ("Push-ups", "4 x 20"),
        ],
    },
    "shoulder": {
        "beginner": [
            ("Dumbbell Shoulder Press", "3 x 12"),
            ("Lateral Raises", "3 x 15"),
        ],
        "intermediate": [
            ("Dumbbell Shoulder Press", "4 x 8"),
            ("Lateral Raises", "3 x 12"),
        ],
        "advanced": [
            ("Dumbbell Shoulder Press", "5 x 5"),
            ("Lateral Raises", "4 x 10"),
        ],
    },
    "legs": {
        "beginner": [("Goblet Squats", "3 x 12"), ("Leg Extensions", "3 x 12")],
        "intermediate": [
            ("Goblet Squats", "4 x 8"),
            ("Leg Extensions", "3 x 10"),
        ],
        "advanced": [
            ("Goblet Squats", "5 x 5"),
            ("Leg Extensions", "4 x 8"),
        ],
    },
    "back": {
        "beginner": [
            ("Lat Pulldowns", "3 x 12"),
            ("Seated Cable Rows", "3 x 12"),
        ],
        "intermediate": [
            ("Lat Pulldowns", "4 x 8"),
            ("Seated Cable Rows", "4 x 8"),
        ],
        "advanced": [
            ("Lat Pulldowns", "5 x 5"),
            ("Seated Cable Rows", "4 x 6-8"),
        ],
    },
    "biceps": {
        "beginner": [("Dumbbell Curls", "3 x 12")],
        "intermediate": [("Dumbbell Curls", "4 x 10")],
        "advanced": [("Dumbbell Curls", "4 x 8")],
    },
    "abs": {
        "beginner": [("Crunches", "3 x 15")],
        "intermediate": [("Crunches", "4 x 20")],
        "advanced": [("Crunches", "5 x 20")],
    },
}

if st.sidebar.button("🚀 Generate Workout Routine"):
  moves = EXERCISES.get(muscle.lower(), {}).get(level.lower(), []).copy()

  # 1. Injury Logic Replacement
  replaced_warning = False
  if injury == "Shoulder" and muscle.lower() == "chest":
    moves = [
        ("Pec Deck Flyes (Joint Safe)", "3 x 12"),
        ("Push-ups", "3 x 10"),
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

  # Select Heatmap URL based on Gender & Selected Muscle
  heatmap_url = (
      MALE_HEATMAPS.get(muscle.lower())
      if gender == "Male"
      else FEMALE_HEATMAPS.get(muscle.lower())
  )

  # Display Each Exercise with 3-Column Layout (Info, Heatmap, GIF)
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
      st.write("**Target Heatmap**")
      st.image(
          heatmap_url,
          caption=f"{muscle} Heatmap ({gender})",
          use_container_width=True,
      )

    with col3:
      st.write("**Exercise Execution**")
      gif_url = GIFS.get(gender, {}).get(
          move,
          "https://media.giphy.com/media/3o7TKRBB3E7yqG3Lq0/giphy.gif",
      )
      st.image(
          gif_url, caption=f"Execution ({gender})", use_container_width=True
      )

    st.divider()

    moves_list.append(move)
    sets_list.append(int(sets_reps.split("x")[0].strip()))

  # Matplotlib Bar Chart
  fig, ax = plt.subplots(figsize=(7, 3))
  ax.barh(moves_list, sets_list, color="#2ecc71")
  ax.set_xlabel("Sets")
  ax.set_title(f"Today's {muscle} Plan Breakdown")
  st.pyplot(fig)
