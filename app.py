import streamlit as st
import time
from src.scraper import JobScraper
from src.llm_service import LLMService
from src.retriever import PortfolioRetriever

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="AtliQ | Cold Email Generator",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        color: white;
    }
    .skill-pill {
        display: inline-block;
        background-color: #f0f2f6;
        border-radius: 15px;
        padding: 5px 15px;
        margin: 5px;
        font-size: 14px;
        color: #31333F;
        border: 1px solid #dcdcdc;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3059/3059997.png", width=80)
    st.title("AtliQ Engine")
    st.markdown("Automated RAG-powered Cold Email Generator.")
    st.divider()
    st.markdown("### How it works:")
    st.markdown("""
    1. **Scrape:** Extracts text from a job URL.
    2. **Analyze:** LLM identifies Role & Skills.
    3. **Search:** Hybrid Search matches your GitHub portfolio.
    4. **Rerank:** Cohere picks the absolute best 2-3 projects.
    5. **Generate:** Groq LLM writes a highly personalized email.
    """)
    st.divider()
    st.caption("© 2026 AtliQ AI Solutions")

# ==========================================
# MAIN APP UI
# ==========================================
st.title("📧 Advanced RAG Cold Email Generator")
st.markdown("Enter a job posting URL below to generate a highly personalized, portfolio-backed cold email.")

# Input Section
job_url = st.text_input("Job Posting URL", placeholder="https://careers.nike.com/...", key="url_input")

generate_btn = st.button("Generate Cold Email")

if generate_btn and job_url:
    try:
        # We use st.status to show step-by-step progress to the user
        with st.status("Initializing AI Pipeline...", expanded=True) as status:
            
            # 1. Scrape
            st.write("🕵️‍♂️ Scraping job posting...")
            page_data = JobScraper.scrape(job_url)
            
            # 2. Extract Details
            st.write("🧠 Extracting job role and required skills...")
            llm_service = LLMService()
            job_details = llm_service.extract_job_details(page_data)
            
            # 3. Retrieve Portfolio
            st.write("🔍 Searching Weaviate and Reranking with Cohere...")
            retriever = PortfolioRetriever()
            portfolio_context = retriever.get_portfolio_context(job_details)
            
            # 4. Generate Email
            st.write("✍️ Drafting personalized email...")
            final_email = llm_service.generate_email(job_details, portfolio_context)
            
            # Close services
            retriever.close()
            
            status.update(label="❇ Generation Complete!", state="complete", expanded=False)

        # ==========================================
        # DISPLAY RESULTS
        # ==========================================
        
        # Create two columns for the output layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Job Analysis")
            st.markdown(f"**Target Role:** `{job_details.get('role', 'N/A')}`")
            
            st.markdown("**Identified Skills:**")
            skills_html = "".join([f'<span class="skill-pill">{skill}</span>' for skill in job_details.get('skills', [])])
            st.markdown(skills_html, unsafe_allow_html=True)
            
            with st.expander("View Matched Portfolio Projects"):
                st.markdown(portfolio_context)
                
        with col2:
            st.subheader("✉️ Generated Email")
            # We use a text area so the user can easily edit/copy it
            st.text_area(
                label="You can edit the email below before sending:",
                value=final_email,
                height=450
            )
            
            st.success("Email generated successfully! Ready to send.")

    except Exception as e:
        st.error(f"🚨 An error occurred: {str(e)}")
        if 'retriever' in locals():
            retriever.close()

elif generate_btn and not job_url:
    st.warning("⚠️ Please enter a Job URL first.")