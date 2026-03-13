from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", temperature=0.1)
response = model.embed_query(
        "What is the capital of France?",
        output_dimensionality=256
    )
print(response[:10])