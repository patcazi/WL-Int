import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from pypdf import PdfReader
import pandas as pd
import json

# 1. Load Environment Variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2. Page Configuration
st.set_page_config(page_title="Westlaw AI Prototype", page_icon="⚖️")

# 3. Initialize OpenAI Client
if not api_key:
    st.error("No API Key found! Check your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

# 3.5. Analysis Function
def analyze_order(text):
    """Analyzes PDF text and extracts structured legal data."""
    try:
        analysis_prompt = f"""Analyze this legal order/opinion and extract the following information in JSON format:

{{
    "case_name": "Case name or title",
    "expert_type": "Type of expert witness (e.g., Medical, Financial, Engineering, etc.)",
    "motion_outcome": "Granted or Denied or Mixed",
    "legal_basis": "Reliability or Qualifications or Procedural or Relevance",
    "one_sentence_summary": "One sentence summarizing the key holding"
}}

Document text:
{text[:8000]}

Respond ONLY with valid JSON matching the structure above. No additional text."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal analyst. Return only valid JSON."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        return json.loads(result_text)
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return {
            "case_name": "Error",
            "expert_type": "Unknown",
            "motion_outcome": "Unknown",
            "legal_basis": "Unknown",
            "one_sentence_summary": f"Error analyzing: {str(e)}"
        }

def generate_judge_profile(cases_data):
    """Generates a strategic judge scouting report based on case patterns."""
    try:
        # Convert cases data into a summary string
        summary = "Dataset of Daubert Rulings:\n\n"
        for idx, case in enumerate(cases_data, 1):
            summary += f"{idx}. {case['case_name']}\n"
            summary += f"   Expert Type: {case['expert_type']}\n"
            summary += f"   Motion Outcome: {case['motion_outcome']}\n"
            summary += f"   Legal Basis: {case['legal_basis']}\n"
            summary += f"   Summary: {case['one_sentence_summary']}\n\n"
        
        # Create the strategic analysis prompt
        strategic_prompt = f"""You are a senior litigation strategist. Review this dataset of {len(cases_data)} Daubert rulings by a single judge. Identify clear patterns:

- Does she have a bias against specific expert types (e.g. Financial vs Medical)?
- Is she strict on Procedure (Rule 26)?
- Is she lenient on Reliability methodology?
- What is the #1 way to get an expert excluded by this judge?

Write a concise, bulleted 'Judge Scouting Report' for a Partner.

CRITICAL - GROUND TRUTH RULES:
1. Source of Truth: Trust the 'Motion Outcome' field above all else.
2. Terminology: A 'Denied' motion means the Expert was ALLOWED to testify (Win for Expert). A 'Granted' motion means the Expert was EXCLUDED (Loss for Expert).
3. Fact Checking: When citing specific cases (e.g., Cleburne, Lowen), verify that your description of the outcome matches the 'Motion Outcome' data provided. Do not claim the judge is strict on a category if the motions were Denied.

{summary}

Provide your analysis in a clear, professional format with specific examples from the cases. Double-check all outcome interpretations before making claims."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior litigation strategist specializing in Daubert motions and expert witness challenges."},
                {"role": "user", "content": strategic_prompt}
            ],
            temperature=0.5
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating judge profile: {str(e)}"

# 4. Session State (Memory)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful legal assistant."}
    ]

if "knowledge_base" not in st.session_state:
    st.session_state["knowledge_base"] = {}

if "model" not in st.session_state:
    st.session_state["model"] = "gpt-4o"

if "analyzed_cases" not in st.session_state:
    st.session_state["analyzed_cases"] = []

# 5. The User Interface
st.title("⚖️ Judge Analytics Dashboard")

# Sidebar for PDF Upload and Model Selection
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model Selector
    st.session_state["model"] = st.selectbox(
        "Select Model",
        options=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        index=0
    )
    
    st.divider()
    
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            
            # Check if file is already in knowledge base
            if filename not in st.session_state["knowledge_base"]:
                # Extract text from PDF
                try:
                    pdf_reader = PdfReader(uploaded_file)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()
                    
                    # Add to knowledge base
                    st.session_state["knowledge_base"][filename] = pdf_text
                    
                    # Analyze the document
                    with st.spinner(f"Analyzing {filename}..."):
                        analysis_result = analyze_order(pdf_text)
                        analysis_result["filename"] = filename
                        st.session_state["analyzed_cases"].append(analysis_result)
                    
                    st.success(f"✅ Analyzed '{filename}' ({len(pdf_reader.pages)} pages)")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
            else:
                st.info(f"'{filename}' is already loaded")
    
    # Display loaded documents
    if st.session_state["knowledge_base"]:
        st.divider()
        st.subheader("📚 Loaded Documents")
        for idx, filename in enumerate(st.session_state["knowledge_base"].keys(), 1):
            char_count = len(st.session_state["knowledge_base"][filename])
            st.text(f"{idx}. {filename} ({char_count:,} chars)")
        
        # Clear all documents button
        if st.button("🗑️ Clear All Documents"):
            st.session_state["knowledge_base"] = {}
            st.session_state["analyzed_cases"] = []
            st.rerun()
    
    # Interactive Filters
    if st.session_state["analyzed_cases"]:
        st.divider()
        st.subheader("🔍 Filters")
        
        # Get unique values for filters
        all_cases = pd.DataFrame(st.session_state["analyzed_cases"])
        
        expert_types = sorted(all_cases["expert_type"].unique().tolist())
        motion_outcomes = sorted(all_cases["motion_outcome"].unique().tolist())
        
        # Expert Type Filter
        st.session_state["selected_experts"] = st.multiselect(
            "Filter by Expert Type",
            options=expert_types,
            default=expert_types,
            key="expert_filter"
        )
        
        # Motion Outcome Filter
        st.session_state["selected_outcomes"] = st.multiselect(
            "Filter by Motion Outcome",
            options=motion_outcomes,
            default=motion_outcomes,
            key="outcome_filter"
        )

# 6. Display Analytics Dashboard
if st.session_state["analyzed_cases"]:
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state["analyzed_cases"])
    
    # Reorder columns for better display
    column_order = ["filename", "case_name", "motion_outcome", "legal_basis", "expert_type", "one_sentence_summary"]
    df = df[column_order]
    
    # Apply Filters from session state
    if "selected_experts" in st.session_state and "selected_outcomes" in st.session_state:
        df_filtered = df[
            (df["expert_type"].isin(st.session_state["selected_experts"])) & 
            (df["motion_outcome"].isin(st.session_state["selected_outcomes"]))
        ]
    else:
        df_filtered = df
    
    # Top Section: KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cases", len(df_filtered))
    with col2:
        granted = len(df_filtered[df_filtered["motion_outcome"] == "Granted"])
        st.metric("Motions Granted", granted)
    with col3:
        denied = len(df_filtered[df_filtered["motion_outcome"] == "Denied"])
        st.metric("Motions Denied", denied)
    
    st.divider()
    
    # Strategy Row: Win Rate by Expert Type & The Kill Zone
    st.subheader("🎯 Strategic Insights")
    
    strategy_left, strategy_right = st.columns(2)
    
    with strategy_left:
        st.markdown("**📊 Win Rate by Expert Type**")
        
        # Calculate win rate by expert type
        if not df_filtered.empty:
            expert_outcomes = df_filtered.groupby(['expert_type', 'motion_outcome']).size().unstack(fill_value=0)
            
            # Calculate percentages
            if not expert_outcomes.empty:
                # Display as stacked bar chart
                st.bar_chart(expert_outcomes)
            else:
                st.info("No data available for current filters")
        else:
            st.info("No data available for current filters")
    
    with strategy_right:
        st.markdown("**☠️ The Kill Zone (Legal Basis)**")
        
        # Calculate legal basis distribution
        if not df_filtered.empty:
            legal_basis_counts = df_filtered["legal_basis"].value_counts()
            
            # Create pie chart data
            if not legal_basis_counts.empty:
                # Display as text with percentages
                total = legal_basis_counts.sum()
                for basis, count in legal_basis_counts.items():
                    percentage = (count / total) * 100
                    st.metric(basis, f"{count} cases", f"{percentage:.1f}%")
            else:
                st.info("No data available for current filters")
        else:
            st.info("No data available for current filters")
    
    # Judge Scouting Report Section
    st.markdown("---")
    
    if st.button("🤖 Generate Judge Scouting Report", type="primary", use_container_width=True):
        with st.spinner("Analyzing judge patterns and generating strategic report..."):
            judge_report = generate_judge_profile(st.session_state["analyzed_cases"])
            st.session_state["judge_report"] = judge_report
    
    # Display the report if it exists
    if "judge_report" in st.session_state:
        st.info("### 📋 Judge Scouting Report")
        st.markdown(st.session_state["judge_report"])
    
    st.divider()
    
    # Data Table Section (Bottom)
    st.subheader("📋 Detailed Case Analysis")
    
    # Export Button
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Analysis as CSV",
        data=csv_data,
        file_name="judge_analytics_export.csv",
        mime="text/csv"
    )
    
    # Enhanced Data Table with Color Highlighting
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "motion_outcome": st.column_config.TextColumn(
                "Ruling",
                help="Motion outcome: Granted, Denied, or Mixed"
            ),
            "filename": st.column_config.TextColumn("Document"),
            "case_name": st.column_config.TextColumn("Case Name"),
            "legal_basis": st.column_config.TextColumn("Legal Basis"),
            "expert_type": st.column_config.TextColumn("Expert Type"),
            "one_sentence_summary": st.column_config.TextColumn("Summary", width="large")
        }
    )
    
    st.divider()

st.header("💬 Ask Follow-Up Questions")

# Display Chat History
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Chat Input & Response Logic
if prompt := st.chat_input("Ask me a question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate Assistant response
    with st.chat_message("assistant"):
        # Prepare messages with PDF context if available
        messages_to_send = st.session_state.messages.copy()
        
        if st.session_state["knowledge_base"]:
            # Combine all PDF texts from knowledge base
            combined_context = "System Context - Document Library:\n\n"
            for filename, text in st.session_state["knowledge_base"].items():
                combined_context += f"=== Document: {filename} ===\n{text}\n\n"
            
            # Insert combined PDF context as system message after the initial system message
            pdf_context_msg = {
                "role": "system",
                "content": combined_context
            }
            messages_to_send.insert(1, pdf_context_msg)
        
        stream = client.chat.completions.create(
            model=st.session_state["model"],
            messages=messages_to_send,
            stream=True,
        )
        response = st.write_stream(stream)
    
    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})