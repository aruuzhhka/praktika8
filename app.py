from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Тапсырманың кешігуін болжау", layout="centered")
st.title("Тапсырманың кеш тапсырылуын болжау")
st.write("Студенттің бұрынғы оқу әрекеттері негізінде келесі жұмыстың мерзімінде орындалуын болжау")
st.info(
    "Оқу демонстрациясы. Модель классификация алгоритмі арқылы оқытылған. "
    "Нәтиже ескерту жасау және уақытты жоспарлауға қолдау көрсету үшін қолданылады."
)

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "late_prediction_model.joblib"
    return joblib.load(path)

bundle = load_model()
model = bundle["model"]
features = bundle["features"]

with st.form("prediction_form"):
    prev_late_pct = st.slider("Алдыңғы жұмыстардың кешігу үлесі (0 - 1 аралығы)", 0.0, 1.0, 0.3, 0.05)
    completed_tasks = st.number_input(
        "Осыған дейін орындалған жұмыстар саны",
        min_value=1, max_value=10, value=5, step=1
    )
    active_days = st.slider("Соңғы екі аптадағы белсенді күндер саны", 1, 14, 7, 1)
    
    clicked = st.form_submit_button("Болжам жасау")

if clicked:
    row = pd.DataFrame([{
        "prev_late_pct": prev_late_pct,
        "completed_tasks": completed_tasks,
        "active_days": active_days
    }])
    
    prediction = int(model.predict(row[features])[0])
    probability = model.predict_proba(row[features])[0][1] # Кешігу ықтималдығы
    
    if prediction == 1:
        st.error(f"⚠️ **Назар аударыңыз:** Бұл тапсырманың кешігу ықтималдығы жоғары ({probability*100:.1f}%).")
        st.warning("Педагогикалық ұсыныс: Студентке мерзімге дейін еске салу жіберіп, уақытты жоспарлауға қолдау көрсету қажет.")
    else:
        st.success(f"✅ **Тамаша:** Тапсырма уақытында тапсырылады деп күтіледі (Кешігу ықтималдығы: {probability*100:.1f}%).")
        
    a, b = st.columns(2)
    a.metric("Кешігу статусы", "Кеш (1)" if prediction == 1 else "Уақытында (0)")
    b.metric("Кешігу ықтималдығы", f"{probability*100:.1f}%")
    
    st.caption("Болжам нақты нәтижеге кепілдік бермейді және тек ескерту сипатында.")