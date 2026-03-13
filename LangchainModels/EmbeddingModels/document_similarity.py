from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import json

load_dotenv()  # Load environment variables from .env file

embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", temperature=0.1)
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)

with open("LangchainModels/data/documents_similarity.json", "r") as f:
    documents = json.loads(f.read())["documents"]

query = "How do solar panels work?"

# Embed the query
query_embedding = embeddings_model.embed_query(
    query,
    output_dimensionality=256
)

# Embed the documents
doc_embeddings = embeddings_model.embed_documents(
    documents,
    output_dimensionality=256
)

# cosine similarity between query and documents
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Get the top 3 most similar documents
top_k = 3
top_k_indices = scores.argsort()[-top_k:][::-1]

# best results
filtered_docs = [documents[i] for i in top_k_indices]

prompt = f"""
You are a helpful assistant. Answer the question based on the following documents:

The question asked by the user is: {query}
The documents are:
{filtered_docs}

Answer the question based on the above documents.
DO NOT use any information outside of the above documents to answer the question.

If you don't know the answer, say you don't know.
"""

response = model.invoke(prompt)
print(response.text)