"""
AI Resume Intelligence Platform — Main Entry Point
=====================================================
This page serves as the Executive Dashboard (when a resume exists in the
session) and the Resume Builder / AI Interview Engine (to create one).
Other pages (ATS Analyzer, Skill Gap, Job Matcher, Salary Predictor,
AI Training Center) live in pages/ and read the same st.session_state.resume.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

import config
import database
from data.skills_taxonomy import FIELDS
from engines import resume_generator, ai_client
from engines.ats_engine import analyze_resume
from engines.career_success_predictor import predict_career_success
from utils.styling import inject_css, page_header, kpi_card, glass_card_open, glass_card_close, gauge_chart, donut_chart
from utils.sidebar_chat import render_sidebar_coach
from utils.pdf_generator import build_resume_pdf, TEMPLATES

st.set_page_config(page_title=config.APP_NAME, page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
inject_css()
database.init_db()

if "resume" not in st.session_state:
    st.session_state.resume = None
if "field_selected" not in st.session_state:
    st.session_state.field_selected = None

render_sidebar_coach()

with st.sidebar:
    st.divider()
    st.markdown("##### 🧭 Navigate")
    st.caption("Use the pages list above (ATS Analyzer, Skill Gap, Job Matcher, Salary Predictor, AI Training Center).")
    if st.session_state.resume:
        st.divider()
        if st.button("🔄 Start a new resume", use_container_width=True):
            st.session_state.resume = None
            st.session_state.field_selected = None
            st.rerun()

page_header("AI Resume", "Intelligence", config.APP_TAGLINE)

FIELD_ICONS = {
    "Software Engineering": "💻", "Data Science": "📊", "Artificial Intelligence": "🤖",
    "Machine Learning": "🧮", "Cybersecurity": "🛡️", "Cloud Computing": "☁️",
    "DevOps": "🔁", "Product Management": "🧭", "Business Analyst": "📈",
    "UI/UX Design": "🎨", "Graphic Design": "🖌️", "Digital Marketing": "📣",
    "Finance": "💰", "Accounting": "🧾", "Human Resources": "🤝", "Sales": "🛒",
    "Healthcare": "🩺", "Nursing": "💉", "Mechanical Engineering": "⚙️",
    "Civil Engineering": "🏗️", "Electrical Engineering": "🔌", "Research": "🔬",
    "Education": "📚", "Consulting": "🧩",
}


def field_selection_screen():
    glass_card_open("STEP 1 OF 2")
    st.markdown("### Choose your field")
    st.caption("This loads field-specific questions, skills, and benchmarks for everything downstream.")
    glass_card_close()

    cols = st.columns(4)
    for i, field in enumerate(FIELDS.keys()):
        with cols[i % 4]:
            if st.button(f"{FIELD_ICONS.get(field, '🧩')}  {field}", key=f"field_{field}", use_container_width=True):
                st.session_state.field_selected = field
                st.rerun()


def interview_form_screen(field):
    glass_card_open("STEP 2 OF 2")
    st.markdown(f"### Build your {field} resume")
    st.caption("Fill in what you have — the AI Interview Engine adapts follow-up suggestions as you go.")
    glass_card_close()

    pool = FIELDS[field]

    with st.form("interview_form"):
        st.markdown("##### Personal Information")
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Full Name")
        email = c2.text_input("Email")
        phone = c3.text_input("Phone")
        c4, c5, c6 = st.columns(3)
        linkedin = c4.text_input("LinkedIn URL")
        github = c5.text_input("GitHub URL")
        portfolio = c6.text_input("Portfolio URL")

        st.markdown("##### Education")
        c1, c2 = st.columns(2)
        degree = c1.selectbox("Highest Degree", ["High School", "Associate", "Bachelors", "Masters", "PhD"], index=2)
        university = c2.text_input("University / Institution")
        c3, c4 = st.columns(2)
        grad_year = c3.text_input("Graduation Year")
        cgpa = c4.text_input("CGPA / GPA (optional)")

        st.markdown("##### Experience")
        years_experience = st.slider("Years of Experience", 0, 30, 2)
        company = st.text_input("Most Recent Company")
        position = st.text_input("Position / Title")
        responsibilities = st.text_area("Key Responsibilities (raw notes are fine — AI will polish)")
        exp_achievements = st.text_area("Notable Achievements in This Role")

        st.markdown("##### A Project You're Proud Of")
        project_name = st.text_input("Project Name")
        project_tech = st.text_input("Technologies Used")
        project_desc = st.text_area("What did you build?")
        project_impact = st.text_area("What was the impact / result?")

        st.markdown("##### Skills")
        technical_skills = st.multiselect(
            "Technical Skills", pool["core_skills"] + pool["trending_skills"] + pool["high_paying_skills"],
            help="Start typing — this is built from the live skill taxonomy for this field.",
        )
        soft_skills = st.multiselect("Soft Skills", [
            "Communication", "Leadership", "Teamwork", "Problem Solving", "Adaptability",
            "Time Management", "Critical Thinking", "Creativity", "Collaboration", "Mentorship",
        ])

        st.markdown("##### Certifications, Achievements, Languages")
        certifications = st.multiselect("Certifications", pool["certifications"])
        achievements_text = st.text_area("General Achievements / Awards (one per line)")
        languages = st.multiselect("Languages", ["English", "Spanish", "French", "German", "Mandarin",
                                                    "Hindi", "Tamil", "Arabic", "Portuguese", "Japanese"])

        st.markdown("##### Career Goals & Resume Type")
        career_goals = st.text_area("Where do you want to be in 3-5 years?")
        resume_type = st.radio("Resume Type", ["Fresher", "Professional", "Executive"], index=1, horizontal=True)

        use_ai = st.checkbox(
            f"✨ Enhance with AI ({'live' if ai_client.is_live() else 'offline template mode'})",
            value=True,
        )

        submitted = st.form_submit_button("🚀 Generate Resume", use_container_width=True, type="primary")

    if submitted:
        resume = resume_generator.empty_resume(field)
        resume.update({
            "name": name, "email": email, "phone": phone,
            "linkedin": linkedin, "github": github, "portfolio": portfolio,
            "education_level": degree,
            "education": [{"degree": degree, "university": university, "year": grad_year, "cgpa": cgpa}] if university else [],
            "years_experience": years_experience,
            "experience": [{"company": company, "position": position,
                              "responsibilities": responsibilities, "achievements": exp_achievements}] if company else [],
            "projects": [{"name": project_name, "technologies": project_tech,
                            "description": project_desc, "results": project_impact}] if project_name else [],
            "technical_skills": technical_skills,
            "soft_skills": soft_skills,
            "certifications": certifications,
            "achievements": [a.strip() for a in achievements_text.split("\n") if a.strip()],
            "languages": languages,
            "career_goals": career_goals,
            "resume_type": resume_type,
        })

        if use_ai:
            with st.spinner("Polishing your resume with AI..."):
                resume = resume_generator.enhance_resume(resume)
        elif not resume.get("summary"):
            resume["summary"], _ = resume_generator.generate_summary(resume)

        st.session_state.resume = resume

        if email:
            user_id = database.upsert_user(name, email, field)
            resume_id = database.save_resume(user_id, field, resume_type, resume)
            st.session_state.user_id = user_id
            st.session_state.resume_id = resume_id

        st.rerun()


def dashboard_screen(resume):
    ats_result = analyze_resume(resume)
    st.session_state.ats_result = ats_result
    career = predict_career_success(resume)

    if resume.get("email") and st.session_state.get("resume_id"):
        database.save_ats_score(st.session_state.resume_id, ats_result["rule_based"]["score"], ats_result["rule_based"]["breakdown"])
        database.save_career_prediction(st.session_state.resume_id, career.get("success_probability_pct", 0))

    st.markdown(f"#### Welcome back, {resume.get('name') or 'there'} 👋")
    st.caption(f"{resume.get('field')} · {resume.get('resume_type')} Resume · {resume.get('years_experience')} yrs experience")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("ATS Score", f"{ats_result['rule_based']['score']:.0f}/100", ats_result["label"])
    with k2:
        kpi_card("Career Readiness", f"{career.get('success_probability_pct', 0):.0f}%", "ML-predicted")
    with k3:
        completeness = ats_result["rule_based"]["raw_factors"]["Resume Completeness"] * 100
        kpi_card("Resume Completeness", f"{completeness:.0f}%", "sections filled")
    with k4:
        kpi_card("Skills Listed", f"{ats_result['features']['num_skills']}", "technical + soft")

    st.write("")
    c1, c2 = st.columns([1, 1])
    with c1:
        glass_card_open("ATS SCORE")
        st.plotly_chart(gauge_chart(ats_result["rule_based"]["score"], "Rule-Based ATS Score"), use_container_width=True)
        if ats_result["ml_score"] is not None:
            st.caption(f"ML model's independent estimate: **{ats_result['ml_score']:.1f}/100** (GradientBoosting, trained on 12k synthetic resumes)")
        glass_card_close()
    with c2:
        glass_card_open("CAREER READINESS")
        st.plotly_chart(donut_chart(career.get("success_probability_pct", 0), "Success Probability"), use_container_width=True)
        for label, kind, reason in career.get("drivers", []):
            icon = "🟢" if kind == "high" else "🔴"
            st.caption(f"{icon} **{label}**: {reason}")
        glass_card_close()

    if ats_result["missing_signals"]:
        glass_card_open("QUICK WINS")
        for s in ats_result["missing_signals"]:
            st.markdown(f"- {s}")
        glass_card_close()

    st.write("")
    glass_card_open("EXPORT YOUR RESUME")
    c1, c2 = st.columns([1, 2])
    with c1:
        template = st.selectbox("Template", list(TEMPLATES.keys()))
    with c2:
        st.write("")
        if st.button("📄 Generate PDF", type="primary"):
            out_dir = "/tmp/resume_exports"
            os.makedirs(out_dir, exist_ok=True)
            fname = f"{(resume.get('name') or 'resume').replace(' ', '_')}_{template.replace(' ', '_')}.pdf"
            out_path = os.path.join(out_dir, fname)
            build_resume_pdf(resume, out_path, template=template)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download PDF", f, file_name=fname, mime="application/pdf")
    glass_card_close()

    st.caption("👉 Use the pages in the sidebar for deep ATS breakdowns, Skill Gap Analysis, Job Matching, and Salary Prediction.")


# ---------------- Router ----------------
if st.session_state.resume is not None:
    dashboard_screen(st.session_state.resume)
elif st.session_state.field_selected is not None:
    interview_form_screen(st.session_state.field_selected)
else:
    field_selection_screen()
