from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()  # Load environment variables from .env file

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)


# schema
class Review(TypedDict):
    summary: str
    sentiment: str

class AnnotatedReview(TypedDict):
    summary: Annotated[str, "A brief summary of the movie review."]
    sentiment: Annotated[Literal["Good", "Bad", "Ok"], "The overall sentiment of the review."]
    key_themes: Annotated[list[str], "A list of key themes mentioned in the review."]
    pros: Annotated[Optional[list[str]], "A list of pros mentioned in the review. Can be null if not mentioned."]
    cons: Annotated[Optional[list[str]], "A list of cons mentioned in the review. Can be null if not mentioned."]


structured_model = model.with_structured_output(Review)
annotated_model = model.with_structured_output(AnnotatedReview)

prompt = "Inception is one of those rare films that becomes more rewarding every time you revisit it. Christopher Nolan takes a heist-thriller framework and fuses it with a deeply emotional story about grief, guilt, and the fear of letting go. The central idea, entering dreams to steal or plant thoughts, is ambitious on its own, but what makes the movie truly memorable is how clearly it commits to rules, consequences, and escalating tension. Each dream layer has a purpose, and the way time stretches across levels creates constant pressure that keeps the story moving even when the concepts get complex. Leonardo DiCaprio gives a controlled and vulnerable performance as Cobb, a man who is brilliant at his work but emotionally trapped by his past. His arc gives the film heart, and without that personal conflict, the technical brilliance might have felt cold. The supporting cast is excellent as well: Joseph Gordon-Levitt brings precision and calm, Tom Hardy adds charm and unpredictability, Elliot Page provides curiosity and intelligence as the audience surrogate, and Cillian Murphy delivers a surprisingly tender emotional center to the mission. Visually, the film is still stunning. The folding city, zero-gravity hallway fight, collapsing dream architecture, and mountain fortress sequence all feel practical, tactile, and imaginative in a way that many modern blockbusters struggle to match. Hans Zimmer's score is another major strength, with its heavy, rising motifs amplifying both the spectacle and the melancholy underneath. If there is a criticism, it is that the exposition can feel dense on first watch and some emotional beats are intentionally restrained. Still, those choices fit the movie's puzzle-box design and reward patience. Inception succeeds as both blockbuster entertainment and thoughtful science fiction. It is smart without becoming sterile, emotional without becoming sentimental, and ambitious without losing control. Few films manage that balance."

response = structured_model.invoke(prompt)
annotated_response = annotated_model.invoke(prompt)

print("Structured Output:"  )
print(response)
print("Annotated Output:"  )
print(annotated_response)