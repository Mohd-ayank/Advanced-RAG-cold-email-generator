
from src.scraper import JobScraper
from src.llm_service import LLMService
from src.retriever import PortfolioRetriever

def main():
    job_url = "https://careers.nike.com/lead-software-engineer-aiml-itc/job/R-86384"
    
    # Initialize Services
    llm_service = LLMService()
    retriever = PortfolioRetriever()
    
    try:
        # Step 1: Scrape Web Page
        page_data = JobScraper.scrape(job_url)
        
        # Step 2: Extract JSON data
        job_details = llm_service.extract_job_details(page_data)
        
        # Step 3: Hybrid Search + Reranking
        portfolio_context = retriever.get_portfolio_context(job_details)
        
        # Step 4: Generate Email
        email = llm_service.generate_email(job_details, portfolio_context)
        
        print("\n" + "="*80)
        print(" FINAL COLD EMAIL ")
        print("="*80)
        print(email)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Always close connections
        retriever.close()

if __name__ == "__main__":
    main()
