from langchain_community.document_loaders import WebBaseLoader

class JobScraper:
    @staticmethod
    def scrape(url: str) -> str:
        print(f"Scraping job data from: {url}...")
        try:
            loader = WebBaseLoader(url)
            page_data = loader.load().pop().page_content
            return page_data
        except Exception as e:
            raise Exception(f"Failed to scrape URL: {e}")