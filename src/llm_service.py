from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config import GROQ_API_KEY

class LLMService:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=GROQ_API_KEY
        )

    def extract_job_details(self, page_data: str) -> dict:
        print("Extracting job details using Groq...")
        prompt = PromptTemplate.from_template(
            """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the
        following keys: `role`, `skills` and `description`.
        in skills section, keep all the relevant skills.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):
        """
        )
        chain = prompt | self.llm
        res = chain.invoke(input={'page_data': page_data})
        
        parser = JsonOutputParser()
        return parser.parse(res.content)

    def generate_email(self, job_dict: dict, portfolio_context: str) -> str:
        print("Drafting cold email...")
        prompt = PromptTemplate.from_template("""
        ### JOB DESCRIPTION:
        {description}

        ### INSTRUCTION:
        You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools.
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability,
        process optimization, cost reduction, and heightened overall efficiency.
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ
        in fulfilling their needs.
        Also add these 3 relevant portfolios from the following links to showcase Atliq's portfolio: {portfolio_context}
        Remember you are Mohan, BDE at AtliQ.
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):

        """)
        chain = prompt | self.llm
        res = chain.invoke({
            "role": job_dict.get('role', 'Engineer'),
            "skills": ", ".join(job_dict.get('skills', [])),
            "description": job_dict.get('description', ''),
            "portfolio_context": portfolio_context
        })
        return res.content