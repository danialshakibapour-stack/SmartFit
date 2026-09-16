import streamlit as st
import json
import os

st.set_page_config(page_title="برنامه تمرینی هوشمند", layout="wide")

# مسیر فایل JSON و پوشه ویدیوها
JSON_PATH = os.path.join("VitalAnimations", "Free50", "50gymworkouts.json")
MEDIA_DIR = os.path.join("VitalAnimations", "Free50", "Free50")

@st.cache_data
def load_exercise_data():
    if not os.path.exists(JSON_PATH):
        st.error(f"فایل یافت نشد: {JSON_PATH}")
        return []
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

dataset = load_exercise_data()

st.title("🏋️‍♂️ برنامه تمرینی و حرکات ورزشی")

# نگاشت عضلات فارسی به انگلیسی جهت جستجو در دیتابیس
muscle_mapping = {
    "سینه": "chest",
    "زیربغل / پشت": "back",
    "سرشانه": "shoulders",
    "جلو بازو": "biceps",
    "پشت بازو": "triceps",
    "پا": "legs",
    "شکم": "abs"
}

col1, col2 = st.columns(2)

with col1:
    selected_muscle_fa = st.selectbox("عضله هدف را انتخاب کنید:", list(muscle_mapping.keys()))
    selected_muscle_en = muscle_mapping[selected_muscle_fa]

with col2:
    difficulty = st.selectbox("سطح تمرین:", ["beginner", "intermediate", "advanced"], format_func=lambda x: {"beginner": "مبتدی", "intermediate": "متوسط", "advanced": "پیشرفته"}[x])

# فیلتر حرکات بر اساس عضله و سطح
filtered_exercises = [
    ex for ex in dataset
    if selected_muscle_en in ex.get("target", "").lower() or selected_muscle_en in ex.get("bodyPart", "").lower()
]

# اگر بر اساس سطح پیدا نشد، تمام حرکات آن عضله را نشان می‌دهد
level_filtered = [ex for ex in filtered_exercises if ex.get("difficulty", "").lower() == difficulty]
display_exercises = level_filtered if level_filtered else filtered_exercises

if not display_exercises:
    st.warning("هیچ حرکتی برای این عضله پیدا نشد.")
else:
    st.subheader(f"برنامه پیشنهادی برای {selected_muscle_fa} ({len(display_exercises[:4])} حرکت)")
    
    for idx, ex in enumerate(display_exercises[:4], 1):
        st.markdown("---")
        st.markdown(f"### {idx}. {ex.get('name', '').title()}")
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown(f"**عضله اصلی:** {ex.get('target', 'نامشخص')}")
            st.markdown(f"**عضلات کمکی:** {', '.join(ex.get('secondaryMuscles', []))}")
            st.markdown(f"**تجهیزات:** {ex.get('equipment', 'نامشخص')}")
            st.markdown(f"**سطح:** {ex.get('difficulty', 'نامشخص')}")
            
            st.markdown("**نحوه اجرا:**")
            for step in ex.get("instructions", []):
                st.write(f"- {step}")
                
        with c2:
            # پیدا کردن ویدیو بر اساس ID
            ex_id = ex.get("exerciseId", "")
            video_path = os.path.join(MEDIA_DIR, f"{ex_id}.mp4")
            
            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.info(f"ویدیو برای حرکت کد {ex_id} یافت نشد.")
