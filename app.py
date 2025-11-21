
import re
import pandas as pd
import textstat
import torch
from typing import Dict

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline




# ============= CONFIG =============
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "product_dataset.xlsx"  # upload this file in the same folder when running locally

# ============= HELPERS =============
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.strip()

def compute_keyword_metrics(generated_text: str, keywords_str: str) -> Dict:
    gen = clean_text(generated_text).lower()
    tokens = re.findall(r"\w+", gen)
    total_tokens = max(len(tokens), 1)

    raw_keywords = re.split(r"[|,;]", str(keywords_str))
    keywords = [k.strip().lower() for k in raw_keywords if k.strip()]
    total_keywords = max(len(keywords), 1)

    present = 0
    keyword_token_count = 0

    for kw in keywords:
        if kw in gen:
            present += 1
        keyword_token_count += gen.count(kw)

    coverage = (present / total_keywords) * 100
    density = (keyword_token_count / total_tokens) * 100

    return {
        "keyword_coverage_pct": round(coverage, 2),
        "keyword_density_pct": round(density, 2),
        "total_keywords": total_keywords,
        "present_keywords": present,
    }

def compute_readability(text: str) -> Dict:
    text = clean_text(text)
    if not text:
        return {"flesch_reading_ease": 0.0}
    score = textstat.flesch_reading_ease(text)
    return {"flesch_reading_ease": round(score, 2)}

def check_hallucination(attributes_str: str, generated_text: str) -> Dict:
    attrs_raw = re.split(r"[|,;]", str(attributes_str))
    attrs = [a.strip() for a in attrs_raw if a.strip()]

    gen_low = clean_text(generated_text).lower()

    missing = []
    for a in attrs:
        if a.lower() not in gen_low:
            missing.append(a)

    return {
        "total_attributes": len(attrs),
        "missing_attributes": missing,
        "is_potential_hallucination": len(missing) > 0
    }

def evaluate_description(row, generated_text: str, tone: str) -> Dict:
    kw_metrics = compute_keyword_metrics(generated_text, row["Keywords"])
    read_metrics = compute_readability(generated_text)
    hallucination = check_hallucination(row["Attributes"], generated_text)

    return {
        "readability": read_metrics["flesch_reading_ease"],
        "tone_requested": tone,
        "keyword_coverage_pct": kw_metrics["keyword_coverage_pct"],
        "keyword_density_pct": kw_metrics["keyword_density_pct"],
        "hallucination_warning": hallucination["is_potential_hallucination"],
        "missing_attributes": hallucination["missing_attributes"],
    }

def build_prompt(
    title: str,
    attributes: str,
    keywords: str,
    tone: str = "luxury",
    target_words: int = 100,
    structure: str = "narrative"
) -> str:
    base = f"""
You are an expert e-commerce copywriter.

Task:
Write a product description for an online store.

Constraints:
- Product title: "{title}"
- Product attributes/features: {attributes}
- Use ALL of these SEO keywords naturally in the text: {keywords}
- Target length: around {target_words} words
- Tone: {tone} (make the style clearly {tone})
- Do NOT invent new features that are not in the attributes.

Structure:
"""
    if structure == "bullets":
        structure_text = """\
1. Start with a 1–2 sentence overview.
2. Then add 3–5 concise bullet points highlighting key features.
3. Finish with a short call-to-action sentence."""
    else:
        structure_text = """\
Use a short 2–3 paragraph narrative description and end with a call-to-action sentence."""

    prompt = base + "\n" + structure_text + "\n\nNow write the final description only."
    return prompt.strip()

@st.cache_resource(show_spinner=True)
@st.cache_resource(show_spinner=True)
def load_model_and_data():
    # Load your dataset
        df = pd.read_excel(DATA_PATH)

            # Load tokenizer and model with accelerate (device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"   # accelerate handles devices
            )

                                                    # IMPORTANT: do NOT pass `device=` when using accelerate
        gen_pipeline = pipeline(
                                                                "text-generation",
          model=model,
          tokenizer=tokenizer,
        )

        return df, gen_pipeline


def generate_text(gen_pipeline, prompt: str, target_words: int) -> str:
    max_new_tokens = int(target_words * 2)

    out = gen_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )[0]["generated_text"]

    if prompt in out:
        out = out[len(prompt):].strip()
    return out.strip()


# ============= STREAMLIT UI =============
st.set_page_config(page_title="AI Product Description Generator", layout="wide")

st.title("AI-Driven Product Description Generator")
st.caption("Gemma-2B • Tone & SEO Control • Evaluation Dashboard")

with st.spinner("Loading model & dataset... (first time can take a bit)"):
    df, gen_pipeline = load_model_and_data()

st.sidebar.header("Generation Controls")

tone = st.sidebar.selectbox(
    "Tone",
    ["luxury", "casual", "technical", "playful", "minimal"]
)

structure = st.sidebar.selectbox(
    "Structure",
    ["narrative", "bullets"]
)

target_words = st.sidebar.slider("Target length (words)", 50, 250, 120, step=10)

st.sidebar.markdown("---")
st.sidebar.write("Select a product from dataset or enter manually.")

mode = st.sidebar.radio("Input mode", ["From dataset", "Manual"])

if mode == "From dataset":
      titles = df["Product Title"].tolist()
      selected_title = st.selectbox("Choose product title", titles)
      base_row = df[df["Product Title"] == selected_title].iloc[0]

                  # Title comes from dataset, not editable (you can make it editable if you want)
      title_val = base_row["Product Title"]

      st.subheader("Selected Product (from dataset)")
      st.write(f"**Title:** {title_val}")

                                  # 🔹 NEW: user-editable attributes + keywords (pre-filled from dataset)
      attributes_val = st.text_area(
      "Attributes / Features (you can edit)",
      value=str(base_row["Attributes"])
      )

      keywords_val = st.text_area(
      "SEO Keywords (you can edit)",
          value=str(base_row["Keywords"])
        )

                                                                                      # Use the possibly edited values for evaluation + generation
      row = pd.Series({
        "Product Title": title_val,
        "Attributes": attributes_val,
        "Keywords": keywords_val,
         "Description": base_row.get("Description", "")
       })


else:
    title_val = st.text_input("Product Title")
    attributes_val = st.text_area("Attributes / Features (separate with | or , )")
    keywords_val = st.text_input("SEO Keywords (separate with | or , )")

    row = {
        "Product Title": title_val,
        "Attributes": attributes_val,
        "Keywords": keywords_val,
        "Description": ""
    }
    import pandas as pd
    row = pd.Series(row)

if st.button("Generate Description"):
    if not title_val or not attributes_val or not keywords_val:
        st.error("Please provide title, attributes, and keywords.")
    else:
        with st.spinner("Generating description with Gemma-2B..."):
            prompt = build_prompt(
                title=title_val,
                attributes=attributes_val,
                keywords=keywords_val,
                tone=tone,
                target_words=target_words,
                structure=structure
            )
            gen_text = generate_text(gen_pipeline, prompt, target_words)
            metrics = evaluate_description(row, gen_text, tone=tone)

        st.markdown("## Generated Description")
        st.write(gen_text)

        st.markdown("## Evaluation Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 1. Readability & Tone Alignment")
            st.write(f"**Requested tone:** {metrics['tone_requested']}")
            st.write(f"**Flesch Reading Ease:** {metrics['readability']} (higher = easier to read)")

            st.markdown("### 2. SEO Metrics")
            st.write(f"**Keyword Coverage:** {metrics['keyword_coverage_pct']} %")
            st.write(f"**Keyword Density:** {metrics['keyword_density_pct']} %")

        with col2:
            st.markdown("### 3. Variation Under Prompt Changes")
            st.write("Change tone / structure / target words in the sidebar and regenerate to compare descriptions.")

            st.markdown("### 4. Fluency / Hallucination Check")
            if metrics["hallucination_warning"]:
                st.error("Potential hallucination: some attributes were not mentioned.")
                if metrics["missing_attributes"]:
                    st.write("**Missing attributes:**")
                    for a in metrics["missing_attributes"]:
                        st.write(f"- {a}")
            else:
                st.success("All attributes seem to be covered. No hallucinations detected based on this simple check.")
