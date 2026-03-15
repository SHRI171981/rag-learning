from langchain_core.prompts import PromptTemplate, load_prompt


template = PromptTemplate(
    template = """You are a helpful assistant that helps users to find information about a topic.
You will be given a question and a list of documents that may contain the answer to the question
Question: {question}
Documents:
{documents}
Answer the question based on the documents provided. If you don't know the answer, say you don't know.""",
    input_variables=["question", "documents"],
    validate_template=True
)

question = "What is the capital of France?"
documents = [
    "The capital of France is Paris.",
    "France is a country in Europe.",
    "The Eiffel Tower is located in Paris."
]

prompt = template.format(question=question, documents="\n".join(documents))

template.save("prompts/question_answering_prompt.json")

prompt_template = load_prompt("prompts/question_answering_prompt.json")
print(prompt_template.format(question=question, documents="\n".join(documents)))