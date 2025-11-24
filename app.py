import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from pypdf import PdfReader
import pandas as pd
import json
import altair as alt

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
        analysis_prompt = f"""Analyze this legal order/opinion and extract the following information in JSON format.

STEP 1: DETERMINE MOTION TYPE
- OFFENSIVE: Motion to Strike, Exclude, Limit, or Summary Judgment
- DEFENSIVE: Motion for Leave to Designate, Motion to Add, Motion to Extend Time

STEP 2: CONVERT TO EXPERT STATUS (motion_outcome field)

If Motion is OFFENSIVE:
  - Granted → 'Expert Excluded' (Loss)
  - Denied → 'Expert Admitted' (Win)

If Motion is DEFENSIVE:
  - Granted → 'Expert Admitted' (Win)
  - Denied → 'Expert Excluded' (Loss)

TIE-BREAKER RULES:

1. The 'Net Result' Rule: Do not default to 'Mixed' just because a ruling is split.

2. Classify as 'Expert Admitted' (Win) if:
   - The expert is allowed to testify on their core opinion, even if minor limitations are imposed
   - One minor expert is struck while the main ones stay

3. Classify as 'Expert Excluded' (Loss) if:
   - The expert is struck entirely
   - Their primary opinion (e.g., damages calculation) is excluded, leaving them with nothing useful to say

4. Classify as 'Mixed' ONLY if:
   - There are multiple distinct experts and the judge explicitly splits the baby (e.g., 'Expert A is in, Expert B is out')

JSON STRUCTURE (Extract this information):

{{
    "case_name": "Case name or title",
    "expert_type": "Type of expert witness (e.g., Medical, Financial, Engineering, etc.)",
    "motion_type": "Offensive or Defensive",
    "motion_outcome": "Expert Admitted or Expert Excluded or Mixed",
    "legal_basis": "Reliability or Qualifications or Procedural or Relevance",
    "one_sentence_summary": "One sentence summarizing the key holding",
    "key_quote": "Extract 1-2 verbatim sentences that explain WHY the judge ruled this way (the legal reasoning or procedural failure) AND the final ruling itself. Do not just extract the words 'Motion Granted'. Find the sentence that connects the reason to the result.",
    "page_number": "The page number where this quote appears (based on the '--- PAGE X ---' markers)"
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
            temperature=0.0
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
            "motion_type": "Unknown",
            "motion_outcome": "Unknown",
            "legal_basis": "Unknown",
            "one_sentence_summary": f"Error analyzing: {str(e)}",
            "key_quote": "N/A",
            "page_number": "N/A"
        }

def generate_judge_profile(cases_data):
    """Generates a strategic judge scouting report based on case patterns."""
    try:
        # Convert cases data into a summary string
        summary = "Dataset of Daubert Rulings:\n\n"
        for idx, case in enumerate(cases_data, 1):
            summary += f"{idx}. {case['case_name']}\n"
            summary += f"   Expert Type: {case['expert_type']}\n"
            summary += f"   Motion Type: {case.get('motion_type', 'Unknown')}\n"
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

CRITICAL - INTERPRETATION GUIDE:

The 'Motion Outcome' field uses standardized labels:
  - 'Expert Admitted' = The expert was allowed to testify (WIN for the party offering the expert)
  - 'Expert Excluded' = The expert was struck/excluded (LOSS for the party offering the expert)
  - 'Mixed' = Split decision across multiple experts or partial exclusion

When analyzing patterns:
  - Count 'Expert Admitted' outcomes as the judge being LENIENT
  - Count 'Expert Excluded' outcomes as the judge being STRICT
  - Look for patterns by Expert Type, Legal Basis, and overall tendencies

{summary}

Provide your analysis in a clear, professional format with specific examples from the cases. Focus on identifying which types of experts and which legal challenges this judge is most likely to accept or reject."""

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
st.title("⚖️ Know Your Judge: Expert Survival Analytics")
st.caption("Strategic Intelligence: Will your expert testify before Judge Boyle?")

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
                    for i, page in enumerate(pdf_reader.pages):
                        pdf_text += f"\n--- PAGE {i+1} ---\n" + page.extract_text()
                    
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
    
    # Create citation column by combining key_quote and page_number
    df['citation'] = '"' + df['key_quote'].astype(str) + '" (p. ' + df['page_number'].astype(str) + ')'
    
    # Reorder columns for better display
    column_order = ["filename", "case_name", "motion_type", "motion_outcome", "legal_basis", "expert_type", "citation", "one_sentence_summary"]
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
        admitted = len(df_filtered[df_filtered["motion_outcome"] == "Expert Admitted"])
        st.metric("Experts Admitted", admitted)
    with col3:
        excluded = len(df_filtered[df_filtered["motion_outcome"] == "Expert Excluded"])
        st.metric("Experts Excluded", excluded)
    
    st.divider()
    
    # Strategy Row: Win Rate by Expert Type & The Kill Zone
    st.subheader("🎯 Strategic Insights")
    
    # Chart 1: Win Rate by Expert Type (Full Width)
    st.markdown("**📊 Win Rate by Expert Type**")
    st.caption("ℹ️ Labels indicate: Excluded / Total Challenges (Exclusion Rate %)")
    
    if not df_filtered.empty:
        # Group data by expert_type and motion_outcome
        expert_chart_data = df_filtered.groupby(['expert_type', 'motion_outcome']).size().reset_index(name='count')
        
        if not expert_chart_data.empty:
            # Prepare data for labeling - calculate exclusion rates
            expert_stats = df_filtered.groupby('expert_type').agg(
                total_count=('expert_type', 'size')
            ).reset_index()
            
            # Calculate excluded count for each expert type
            excluded_df = df_filtered[df_filtered['motion_outcome'] == 'Expert Excluded'].groupby('expert_type').size().reset_index(name='excluded_count')
            expert_stats = expert_stats.merge(excluded_df, on='expert_type', how='left')
            expert_stats['excluded_count'] = expert_stats['excluded_count'].fillna(0).astype(int)
            
            # Calculate exclusion rate and create label text
            expert_stats['exclusion_rate'] = (expert_stats['excluded_count'] / expert_stats['total_count'] * 100).round(1)
            expert_stats['label_text'] = expert_stats.apply(
                lambda row: f"{row['excluded_count']}/{row['total_count']} ({row['exclusion_rate']:.1f}%)" if row['total_count'] > 0 else "N/A",
                axis=1
            )
            
            # Calculate max count for X-axis domain padding
            max_count = expert_stats['total_count'].max()
            
            # Layer 1: Horizontal stacked bar chart
            bars = alt.Chart(expert_chart_data).mark_bar().encode(
                y=alt.Y('expert_type:N', 
                       title='Expert Type',
                       sort=alt.EncodingSortField(field='count', op='sum', order='descending'),
                       axis=alt.Axis(labels=True)),
                x=alt.X('sum(count):Q', 
                       title='Number of Cases',
                       scale=alt.Scale(domain=[0, max_count * 1.3])),
                color=alt.Color('motion_outcome:N', 
                               title='Outcome',
                               scale=alt.Scale(domain=['Expert Admitted', 'Expert Excluded', 'Mixed'],
                                              range=['#1f77b4', '#d62728', '#ff7f0e'])),
                tooltip=[
                    alt.Tooltip('expert_type:N', title='Expert Type'),
                    alt.Tooltip('motion_outcome:N', title='Outcome'),
                    alt.Tooltip('count:Q', title='Count')
                ]
            )
            
            # Layer 2: Text labels on right side of bars
            text = alt.Chart(expert_stats).mark_text(
                align='left',
                dx=5,
                color='black',
                fontWeight='bold',
                fontSize=12
            ).encode(
                y=alt.Y('expert_type:N',
                       sort=alt.EncodingSortField(field='total_count', order='descending')),
                x=alt.X('total_count:Q'),
                text='label_text:N'
            )
            
            # Combine layers
            expert_chart = alt.layer(bars, text).properties(
                title='Win Rate by Expert Type',
                height=500
            )
            
            st.altair_chart(expert_chart, use_container_width=True)
        else:
            st.info("No data available for current filters")
    else:
        st.info("No data available for current filters")
    
    st.divider()
    
    # Chart 2: The Kill Zone (Full Width)
    st.markdown("**☠️ The Kill Zone (Legal Basis)**")
    st.caption("ℹ️ Labels indicate: Excluded / Total Challenges (Exclusion Rate %)")
    
    if not df_filtered.empty:
        # Group data by legal_basis and motion_outcome
        chart_data = df_filtered.groupby(['legal_basis', 'motion_outcome']).size().reset_index(name='count')
        
        if not chart_data.empty:
            # Prepare data for labeling - calculate exclusion rates
            basis_stats = df_filtered.groupby('legal_basis').agg(
                total_count=('legal_basis', 'size')
            ).reset_index()
            
            # Calculate excluded count for each legal basis
            excluded_df = df_filtered[df_filtered['motion_outcome'] == 'Expert Excluded'].groupby('legal_basis').size().reset_index(name='excluded_count')
            basis_stats = basis_stats.merge(excluded_df, on='legal_basis', how='left')
            basis_stats['excluded_count'] = basis_stats['excluded_count'].fillna(0).astype(int)
            
            # Calculate exclusion rate and create label text
            basis_stats['exclusion_rate'] = (basis_stats['excluded_count'] / basis_stats['total_count'] * 100).round(1)
            basis_stats['label_text'] = basis_stats.apply(
                lambda row: f"{row['excluded_count']}/{row['total_count']} ({row['exclusion_rate']:.1f}%)" if row['total_count'] > 0 else "N/A",
                axis=1
            )
            
            # Calculate max count for X-axis domain padding
            max_count = basis_stats['total_count'].max()
            
            # Layer 1: Horizontal stacked bar chart
            bars = alt.Chart(chart_data).mark_bar().encode(
                y=alt.Y('legal_basis:N', 
                       title='Legal Basis',
                       sort=alt.EncodingSortField(field='count', op='sum', order='descending'),
                       axis=alt.Axis(labels=True)),
                x=alt.X('sum(count):Q', 
                       title='Number of Cases',
                       scale=alt.Scale(domain=[0, max_count * 1.3])),
                color=alt.Color('motion_outcome:N', 
                               title='Outcome',
                               scale=alt.Scale(domain=['Expert Admitted', 'Expert Excluded', 'Mixed'],
                                              range=['#1f77b4', '#d62728', '#ff7f0e'])),
                tooltip=[
                    alt.Tooltip('legal_basis:N', title='Legal Basis'),
                    alt.Tooltip('motion_outcome:N', title='Outcome'),
                    alt.Tooltip('count:Q', title='Count')
                ]
            )
            
            # Layer 2: Text labels on right side of bars
            text = alt.Chart(basis_stats).mark_text(
                align='left',
                dx=5,
                color='black',
                fontWeight='bold',
                fontSize=12
            ).encode(
                y=alt.Y('legal_basis:N',
                       sort=alt.EncodingSortField(field='total_count', order='descending')),
                x=alt.X('total_count:Q'),
                text='label_text:N'
            )
            
            # Combine layers
            chart = alt.layer(bars, text).properties(
                title='Why Experts Get Challenged (Volume vs. Success)',
                height=500
            )
            
            st.altair_chart(chart, use_container_width=True)
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
            "filename": st.column_config.TextColumn("Document"),
            "case_name": st.column_config.TextColumn("Case Name"),
            "motion_type": st.column_config.TextColumn(
                "Motion Type",
                help="Offensive (Strike/Exclude) or Defensive (Motion for Leave)"
            ),
            "motion_outcome": st.column_config.TextColumn(
                "Expert Status",
                help="Expert Admitted, Expert Excluded, or Mixed"
            ),
            "legal_basis": st.column_config.TextColumn("Legal Basis"),
            "expert_type": st.column_config.TextColumn("Expert Type"),
            "citation": st.column_config.TextColumn("Judge's Ruling (Source)", width="large"),
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