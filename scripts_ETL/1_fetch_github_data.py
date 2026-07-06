""" 
Pulls the relevant repos from GitHub """

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN is missing in the .env file!")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# The list of technologies to search
QUERIES = [
    "react", "angular", "vue", "nextjs", "nodejs", "express", "nestjs",
    "django", "flask", "fastapi", "spring boot", "java", "kotlin", "android",
    "swift ios", "flutter", "react native", "mongodb", "postgresql", "mysql",
    "redis", "docker", "kubernetes", "tensorflow", "pytorch", "langchain",
    "llamaindex", "weaviate", "pinecone", "milvus", "rag", "llm", "generative ai"
]

def fetch_github_repos():
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    output_file = "data/github_repositories.csv"
    
    dataset = []
    seen_urls = set()

    for query in QUERIES:
        print(f"Fetching repositories for: {query}...")
        
        # Grab top 50 repos per tech stack
        url = f"https://api.github.com/search/repositories?q={query}&sort=stars&per_page=50&page=1"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code != 200:
            print(f"Error fetching {query}: {response.json()}")
            continue
            
        repos = response.json().get("items", [])
        
        for repo in repos:
            html_url = repo["html_url"]
            
            # Remove Duplicates
            if html_url in seen_urls:
                continue
            seen_urls.add(html_url)
            
            owner = repo["owner"]["login"]
            repo_name = repo["name"]
            
            # Download README in raw text format
            readme_url = f"https://api.github.com/repos/{owner}/{repo_name}/readme"
            readme_headers = HEADERS.copy()
            readme_headers["Accept"] = "application/vnd.github.v3.raw"
            
            readme_res = requests.get(readme_url, headers=readme_headers)
            readme_text = readme_res.text[:1000] if readme_res.status_code == 200 else "No README available."
            
            dataset.append({
                "Project": repo_name,
                "Description": repo.get("description", "No description"),
                "README Summary": readme_text.replace("\n", " "), # Clean up newlines
                "Tech Stack": repo.get("language", "Unknown"),
                "Topics": ", ".join(repo.get("topics", [])),
                "Stars": repo["stargazers_count"],
                "GitHub URL": html_url,
                "Owner": owner
            })
        
        # Pause for 2 seconds to respect GitHub API rate limits
        time.sleep(2) 

    # Save the final dataset to a CSV file
    df = pd.DataFrame(github_repositories)
    df.to_csv(output_file, index=False)
    print(f"\n Successfully saved {len(github_repositories)} unique repositories to {output_file}!")

if __name__ == "__main__":
    fetch_github_repos()
