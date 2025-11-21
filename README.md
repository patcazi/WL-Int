# Westlaw Judge Analytics Dashboard (POC)

## 1. Product Philosophy

Instead of building a simple document summarizer, I built a **Strategic Analytics Dashboard**. The goal was to move from 'reading' to 'reasoning'—helping a litigator answer the core question: *"Will Judge Boyle let my expert testify?"*

## 2. Technical Architecture

  * **Stack:** Python, Streamlit (for rapid UI), Pandas (for analytics).
  * **AI Engine:** OpenAI GPT-4o via Custom System Prompts.
  * **Extraction Pipeline:**
    1.  **Ingest:** `pypdf` extracts raw text from PDFs.
    2.  **Structure:** An LLM agent converts unstructured text into a strict JSON schema (`motion_outcome`, `legal_basis`, `expert_type`).
    3.  **Synthesize:** A second 'Senior Strategist' agent reviews the aggregated dataset to generate the 'Scouting Report.'

## 3. Key Features

  * **The 'Kill Zone' Analysis:** Visualizes the legal basis for exclusions (e.g., Procedure vs. Reliability).
  * **Expert Type Heatmap:** A stacked bar chart showing win/loss rates by expert category (Medical vs. Financial).
  * **Automated Scouting Report:** A 'One-Click' strategy memo that synthesizes the 19 rulings into actionable advice.
  * **Interactive Filters:** Allows the user to slice data by expert type or outcome.

## 4. Key Findings (Derived from the Tool)

  * **Procedural Hardliner:** The #1 predictor of exclusion is **Procedural Failure** (Rule 26 violations). Judge Boyle is unforgiving on missed deadlines.
  * **Scientific Liberal:** She allows the vast majority of Medical/Scientific experts, often citing that 'shaky' evidence is for the jury to weigh.
  * **Financial Skepticism:** She appears significantly stricter on Financial/Damages experts, often citing 'speculation' as grounds for exclusion.

