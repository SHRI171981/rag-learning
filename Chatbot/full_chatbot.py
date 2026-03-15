from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)

chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    user_input = input("User: ")
    if user_input == "exit":
        print("Exiting chatbot. Goodbye!")
        break
    chat_history.append(HumanMessage(content=user_input))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.text))
    print(f"AI: {response.text}")

print(chat_history)