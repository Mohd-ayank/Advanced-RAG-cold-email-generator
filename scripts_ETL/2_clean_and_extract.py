
""" Cleans the text and uses Groq to extract structured JSON) """


import os
import re
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# Load environment variables (API Keys)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def clean_readme(readme):
    """Clean GitHub README while preserving semantic information."""
    if pd.isna(readme):
        return "No README available."

    text = str(readme)
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', ' ', text)
    text = re.sub(r'!\[[^\]]*\]\([^)]+\)', ' ', text)
    text = re.sub(r'<img[^>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://img\.shields\.io\S+', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'[-*_]{3,}', ' ', text)

    commands = [
        r'pip install.*', r'npm install.*', r'npm i.*',
        r'git clone.*', r'cargo install.*', r'composer install.*', r'yarn install.*'
    ]
    for cmd in commands:
        text = re.sub(cmd, ' ', text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_github_dataset(input_csv, output_csv):
    """Handles missing values and cleans the raw dataset."""
    print(f"Reading raw data from {input_csv}...")
    df = pd.read_csv(input_csv)

    df.fillna({
        "Description": "No description available.",
        "README Summary": "No README available.",
        "Tech Stack": "Unknown",
        "Topics": "Unknown"
    }, inplace=True)

    df["Description"] = df["Description"].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    df["README Summary"] = df["README Summary"].apply(clean_readme)
    df["Tech Stack"] = df["Tech Stack"].astype(str).str.strip()
    df["Topics"] = df["Topics"].astype(str).str.strip()
    df["Project"] = df["Project"].astype(str).str.strip()
    df["GitHub URL"] = df["GitHub URL"].astype(str).str.strip()

    df.to_csv(output_csv, index=False)
    print(f"Cleaned dataset saved to {output_csv}")
    return df


def main():
    # Define File Paths
    RAW_CSV = "data/github_repositories.csv"
    CLEANED_CSV = "data/cleaned_github_repos.csv"
    FINAL_JSON_CSV = "data/github_portfolio_dataset.csv"

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(RAW_CSV):
        print(f"Error: Please place your raw dataset at '{RAW_CSV}' before running.")
        return

    # 1. Clean the raw data
    df = clean_github_dataset(RAW_CSV, CLEANED_CSV)

    # 2. Setup LLM
    print("\nInitializing Groq LLM for JSON Extraction...")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = PromptTemplate.from_template(
        """
        You are an expert software engineer and technical recruiter.
        Extract ONLY the information useful for matching this repository with software engineering job descriptions.
        Return ONLY valid JSON.
        
        JSON Schema:
        {{
          "project": "",
          "summary": "",
          "skills": [],
          "key_features": [],
          "concepts": [],
          "project_type": "",
          "github_url": ""
        }}
        
        Repository Information:
        Project: {project}
        Description: {description}
        Tech Stack: {tech_stack}
        Topics: {topics}
        README: {readme}
        GitHub URL: {github_url}
        """
    )

    chain = prompt | llm
    json_parser = JsonOutputParser()

    def process_repository(row):
        response = chain.invoke({
            "project": row["Project"],
            "description": row["Description"],
            "tech_stack": row["Tech Stack"],
            "topics": row["Topics"],
            "readme": row["README Summary"],
            "github_url": row["GitHub URL"]
        })
        try:
            return json_parser.parse(response.content)
        except OutputParserException:
            return {
                "project": row["Project"], "summary": "", "skills": [],
                "key_features": [], "concepts": [], "project_type": "",
                "github_url": row["GitHub URL"]
            }

    # 3. Resume Support & Processing Loop
    results = []
    try:
        existing = pd.read_csv(FINAL_JSON_CSV)
        results = existing.to_dict("records")
        start = len(results)
        print(f"Resuming from row {start}")
    except FileNotFoundError:
        start = 0

    print(f"\nProcessing {len(df) - start} repositories with LLM...")
    for i in tqdm(range(start, len(df))):
        row = df.iloc[i]
        try:
            result = process_repository(row)
            # Store lists as JSON strings
            result["skills"] = json.dumps(result["skills"])
            result["key_features"] = json.dumps(result["key_features"])
            result["concepts"] = json.dumps(result["concepts"])
            results.append(result)
        except Exception as e:
            print(f"Error at row {i}: {e}")
            results.append({
                "project": row["Project"], "summary": "", "skills": "[]",
                "key_features": "[]", "concepts": "[]", "project_type": "",
                "github_url": row["GitHub URL"]
            })

        # Save every 10 repositories
        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(FINAL_JSON_CSV, index=False)
            time.sleep(1) # Respect API Rate Limits

    # Final Save
    pd.DataFrame(results).to_csv(FINAL_JSON_CSV, index=False)
    print(f"\n✅ Completed Successfully! Final dataset saved to {FINAL_JSON_CSV}")

if __name__ == "__main__":
    main()