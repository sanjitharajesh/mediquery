import streamlit as st
from backend.rag.chain import get_rag_chain

# Page config
st.set_page_config(
    page_title="MediQuery",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize chain
@st.cache_resource
def load_chain():
    return get_rag_chain()

chain = load_chain()

# Drug categories
DRUGS = {
    "ADHD & Focus": [
        ("Adderall", "Stimulant medication that helps improve focus and attention"),
        ("Ritalin", "Commonly prescribed stimulant for ADHD and narcolepsy"),
        ("Vyvanse", "Long-acting ADHD medication"),
        ("Strattera", "Non-stimulant ADHD treatment"),
    ],
    "Depression & Anxiety": [
        ("Prozac", "Antidepressant for depression, anxiety, and OCD"),
        ("Zoloft", "Widely used antidepressant"),
        ("Lexapro", "Common antidepressant for depression and anxiety"),
        ("Wellbutrin", "Antidepressant that also helps with smoking cessation"),
    ],
    "Heart & Cholesterol": [
        ("Lipitor", "Statin that lowers cholesterol"),
        ("Atorvastatin", "Generic form of Lipitor"),
        ("Lisinopril", "Blood pressure medication"),
        ("Metoprolol", "Beta blocker for blood pressure"),
    ],
    "Diabetes": [
        ("Metformin", "First-line diabetes medication"),
        ("Ozempic", "Injectable for diabetes and weight management"),
        ("Insulin", "Hormone therapy for blood sugar control"),
    ],
    "Pain Relief": [
        ("Ibuprofen", "Anti-inflammatory pain reliever (Advil, Motrin)"),
        ("Naproxen", "Longer-lasting pain reliever"),
        ("Acetaminophen", "Pain and fever reducer (Tylenol)"),
    ],
}

AGE_GROUPS = {
    "Child (under 12)": "Children require different doses. Consult a pediatrician.",
    "Teen (13-17)": "May need special monitoring for growth and mental health.",
    "Adult (18-64)": "Standard dosing. Women should discuss pregnancy plans.",
    "Senior (65+)": "May need lower doses and higher risk for side effects.",
}

# Custom CSS - FULL CONTROL
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Lora:wght@400;500;600&display=swap');

/* Remove Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Background */
.stApp {
    background: #dbeafe !important;
}

:root {
    --panel-bg: #d5e3ef;
    --panel-border: #adc3d6;
    --panel-bg-hover: #cadbec;
    --panel-text: #1e3a5f;
}

/* Main container */
.main .block-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0rem 3rem 2rem 3rem !important;
}

/* Fonts */
* {
    font-family: 'DM Sans', sans-serif !important;
}

/* Title */
h1 {
    color: #1c2e4a !important;
    font-size: 4rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    margin-top: -0.6rem !important;
    margin-bottom: 0.5rem !important;
    padding-top: 0 !important;
}

/* Subtitle */
.subtitle {
    font-family: 'Lora', serif !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    color: #082052 !important;
    line-height: 1.6 !important;
    margin-bottom: 3rem !important;
    margin-top: 0 !important;
}

/* Section headers */
.section-header {
    color: #1f2937 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
}

/* Labels */
label, .stSelectbox label {
    color: #1f2937 !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}

/* DROPDOWNS */
.stSelectbox > div > div {
    background-color: var(--panel-bg) !important;
    color: var(--panel-text) !important;
    border: 1.5px solid var(--panel-border) !important;
    border-radius: 0.5rem !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    max-width: 240px !important;
}
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] > div {
    background-color: var(--panel-bg) !important;
    color: var(--panel-text) !important;
    border-color: var(--panel-border) !important;
    font-size: 0.85rem !important;
    max-width: 240px !important;
}
.stSelectbox {
    max-width: 240px !important;
}

/* Dropdown menu items */
[role="listbox"] {
    background-color: var(--panel-bg) !important;
    font-size: 0.85rem !important;
    max-width: 240px !important;
}
[role="option"] {
    background-color: var(--panel-bg) !important;
    color: var(--panel-text) !important;
    font-size: 0.85rem !important;
}
[role="option"]:hover {
    background-color: var(--panel-bg-hover) !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1.5px solid #93c5fd !important;
    border-radius: 0.5rem !important;
}

/* Answer box */
.answer-box {
    background-color: var(--panel-bg) !important;
    color: var(--panel-text) !important;
    padding: 1.5rem !important;
    border-radius: 0.5rem !important;
    border: 1.5px solid var(--panel-border) !important;
    min-height: 400px !important;
    white-space: pre-wrap !important;
    line-height: 1.7 !important;
}

/* Answer label */
.answer-label {
    color: #1c2e4a !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.5rem !important;
    margin-top: 1.5rem !important;
    display: block !important;
}

/* Buttons - base styling */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 0.5rem !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1.5rem !important;
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* ✅ Equal height for Get Answer / Clear buttons */
.mq-actions .stButton > button {
    height: 44px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Drug cards */
.drug-card {
    background: white;
    padding: 0.75rem;
    border-radius: 0.5rem;
    border: 1px solid #e5e7eb;
    margin-bottom: 0.75rem;
}
.drug-card strong { font-size: 0.875rem !important; }
.drug-card span { font-size: 0.8rem !important; }

/* Dividers */
hr {
    border-color: #93c5fd !important;
    margin: 1.5rem 0 !important;
    max-width: 240px !important;
}

/* =========================================================
   ✅ MAIN FIX: Target the *exact* main columns row by :has(textarea)
   This avoids nested column blocks (buttons/common questions)
   ========================================================= */

/* Main 2-col row gap reduced (pushes right panel left) */
div[data-testid="stHorizontalBlock"]:has(textarea) {
    gap: 0.75rem !important;
    align-items: flex-start !important;
}

/* Left column fixed width */
div[data-testid="stHorizontalBlock"]:has(textarea) > div[data-testid="column"]:nth-child(1) {
    flex: 0 0 280px !important;
    width: 280px !important;
    max-width: 280px !important;
}

/* Right column narrower + pushed left */
div[data-testid="stHorizontalBlock"]:has(textarea) > div[data-testid="column"]:nth-child(2) {
    flex: 0 0 560px !important;
    width: 560px !important;
    max-width: 560px !important;
    margin-left: -80px !important;
}

/* Make right-side inputs fill the new width */
div[data-testid="stHorizontalBlock"]:has(textarea) > div[data-testid="column"]:nth-child(2) textarea,
div[data-testid="stHorizontalBlock"]:has(textarea) > div[data-testid="column"]:nth-child(2) [data-testid="stTextArea"] {
    width: 100% !important;
    max-width: 100% !important;
}
/* Footer */
.footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    text-align: center;
    font-size: 0.8rem;
    color: #475569;
    opacity: 0.85;
}

.footer span {
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'answer' not in st.session_state:
    st.session_state.answer = ""

# Header
st.markdown("# MediQuery")
st.markdown(
    '<p class="subtitle">AI-powered FDA drug information assistant designed to curb medical hallucination, utilizing only evidence-based official drug labels</p>',
    unsafe_allow_html=True
)

# Layout
col1, col2 = st.columns([1, 1.8], gap="small")

with col1:
    st.markdown('<p class="section-header">Personal Information</p>', unsafe_allow_html=True)

    age = st.selectbox(
        "Age Group",
        ["Select"] + list(AGE_GROUPS.keys()),
        help="Dosing varies by age"
    )

    gender = st.selectbox(
        "Gender",
        ["Select", "Male", "Female", "Other"],
        help="Gender-specific warnings"
    )

    # Conditional pregnancy
    if gender == "Female" and age == "Adult (18-64)":
        pregnant = st.selectbox(
            "Pregnancy Status",
            ["Select", "Yes", "No", "Planning to conceive"],
            help="Critical for medication safety"
        )
    else:
        pregnant = "Select"

    conditions = st.selectbox(
        "Pre-existing Conditions",
        ["Select", "None", "Kidney disease", "Liver disease", "Heart disease", "Diabetes", "High blood pressure", "Multiple conditions"],
        help="Drug interactions and contraindications"
    )

    taking_meds = st.selectbox(
        "Current Medications",
        ["Select", "No medications", "1-2 medications", "3-5 medications", "6+ medications", "Prefer not to say"],
        help="Drug-drug interactions"
    )

    st.markdown("---")
    st.markdown('<p class="section-header">Browse by Category</p>', unsafe_allow_html=True)

    category = st.selectbox(
        "Medical Concern",
        ["Select"] + list(DRUGS.keys()),
        help="Explore medications by condition"
    )

    if category and category != "Select":
        for name, desc in DRUGS.get(category, []):
            st.markdown(f"""
            <div class="drug-card">
                <strong style="color: #1f2937;">{name}</strong><br>
                <span style="color: #6b7280;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown('<p class="section-header">Ask a Question</p>', unsafe_allow_html=True)

    question = st.text_area(
        "Your Question",
        placeholder="Example: What are the side effects of Lipitor?",
        height=80
    )

    # ✅ Equal-size action buttons (equal width + equal height)
    st.markdown('<div class="mq-actions">', unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2, gap="small")

    with col_btn1:
        if st.button("Get Answer", type="primary", use_container_width=True):
            if question:
                with st.spinner("Searching FDA documents..."):
                    ans = chain.invoke(question, verbose=False)

                    ans = ans.replace("WARNING:", "## WARNING\n")
                    ans = ans.replace("PRECAUTIONS:", "## PRECAUTIONS\n")
                    ans = ans.replace("POTENTIAL ADVERSE EFFECTS:", "## POTENTIAL SIDE EFFECTS\n")

                    notes = []
                    if age != "Select":
                        note = AGE_GROUPS.get(age, "")
                        if note and any(w in question.lower() for w in ['dose', 'dosage', 'safety', 'take', 'how']):
                            notes.append(f"**Age Consideration ({age}):** {note}")

                    if gender == "Female" and pregnant == "Yes":
                        preg_drugs = ['accutane', 'isotretinoin', 'lisinopril', 'atorvastatin', 'lipitor', 'metformin']
                        if any(d in question.lower() for d in preg_drugs):
                            notes.append(
                                "**PREGNANCY ALERT:** This medication may have pregnancy-related warnings. "
                                "Review the WARNING section above carefully and consult your doctor immediately."
                            )

                    if conditions in ["Kidney disease", "Liver disease", "Multiple conditions"]:
                        notes.append(
                            f"**Medical Condition Alert ({conditions}):** Some medications require dose adjustments "
                            f"or are contraindicated with {conditions.lower()}."
                        )

                    if taking_meds in ["3-5 medications", "6+ medications"]:
                        notes.append(
                            "**Drug Interaction Alert:** You are taking multiple medications. "
                            "Check for potential drug-drug interactions with your pharmacist."
                        )

                    if notes:
                        ans += "\n\n---\n\n**Personalized Notes:**\n\n" + "\n\n".join(notes)

                    ans += "\n\n---\n\n**Source:** FDA-approved drug labels and prescribing information"
                    ans += "\n\n**Medical Disclaimer:** This information is for educational purposes only. Always consult your healthcare provider before starting, stopping, or changing any medication."

                    st.session_state.answer = ans

    with col_btn2:
        if st.button("Clear", use_container_width=True):
            st.session_state.answer = ""
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<p style="color: #1f2937; font-weight: 600; font-size: 0.9375rem; margin-top: 1rem; margin-bottom: 0.5rem;">Common Questions (click to use):</p>',
        unsafe_allow_html=True
    )

    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        if st.button("What is this medication used for?", key="ex1"):
            question = "What is this medication used for?"
        if st.button("What are the side effects?", key="ex2"):
            question = "What are the side effects?"
        if st.button("Can I take this during pregnancy?", key="ex3"):
            question = "Can I take this during pregnancy?"

    with col_ex2:
        if st.button("How should I take this medication?", key="ex4"):
            question = "How should I take this medication?"
        if st.button("What drugs interact with this?", key="ex5"):
            question = "What drugs interact with this?"
        if st.button("When should I avoid this medication?", key="ex6"):
            question = "When should I avoid this medication?"

    st.markdown("---")

    st.markdown('<span class="answer-label">Answer</span>', unsafe_allow_html=True)
    if st.session_state.answer:
        st.markdown(f'<div class="answer-box">{st.session_state.answer}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="answer-box">Your answer will appear here...</div>', unsafe_allow_html=True)
        
st.markdown("""
<div class="footer">
    © 2026 Sanjitha Rajesh · Built with <span>Streamlit</span>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    pass
