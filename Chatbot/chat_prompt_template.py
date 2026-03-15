from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain what is {topic} in 20words."),
])

prompt = chat_template.format(domain="AI", topic="machine learning")

print(prompt)