from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json

load_dotenv()

prompt = PromptTemplate(
    template="Write a few words about the topic: {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)

chain = prompt | model | parser
chain.get_graph().print_ascii()

result = chain.invoke("Artificial Intelligence")

print(result)
