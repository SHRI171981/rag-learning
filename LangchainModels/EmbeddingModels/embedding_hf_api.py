from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

response = model.embed_query(
    "What is the capital of France?"
)
print(response[:10])