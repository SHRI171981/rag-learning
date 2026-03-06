from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)

response = model.invoke("Write a small story about a dog and a cat. Keep it under 30 words")
print(response.text)