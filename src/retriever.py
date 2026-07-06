import weaviate
import cohere
from weaviate.auth import AuthApiKey
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import WEAVIATE_URL, WEAVIATE_API_KEY, HF_TOKEN, COHERE_API_KEY

class PortfolioRetriever:
    def __init__(self):
        print("Initializing Weaviate & Cohere Clients...")
        self.weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
            headers={"X-HuggingFace-Api-Key": HF_TOKEN}
        )
        self.cohere_client = cohere.Client(COHERE_API_KEY)
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def get_portfolio_context(self, job_dict: dict) -> str:
        # 1. Build Query
        query = f"Role: {job_dict.get('role')} \nSkills: {' '.join(job_dict.get('skills', []))} \nDescription: {job_dict.get('description')}"
        query_embedding = self.embedding_model.embed_query(query)

        # 2. Hybrid Search in Weaviate
        collection = self.weaviate_client.collections.get("Portfolio")
        response = collection.query.hybrid(
            query=query,
            vector=query_embedding,
            alpha=0.5,
            limit=10
        )
        
        if not response.objects:
            return "No relevant portfolio items found."

        # 3. Rerank with Cohere
        docs_to_rerank = [obj.properties["content"] for obj in response.objects]
        rerank_results = self.cohere_client.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=3,
            model="rerank-v3.5"
        )

        # 4. Format Output
        top_portfolios = []
        for result in rerank_results.results:
            best_obj = response.objects[result.index]
            project_name = best_obj.properties["project_name"]
            github_url = best_obj.properties["github_url"]
            content = best_obj.properties["content"]
            
            top_portfolios.append(f"Project: {project_name}\nURL: {github_url}\nDetails: {content}")

        return "\n\n".join(top_portfolios)

    def close(self):
        self.weaviate_client.close()