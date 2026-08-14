import os 
from dotenv import load_dotenv
from pydantic import BaseModel,Field 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_google_genai import ChatGoogleGenerativeAI 

load_dotenv()

class ResearchPlan(BaseModel):
    topic:str = Field(description = "The research topic")
    questions: list[str] = Field(
        description="Important questions that should be investigated"
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research planning assistant."
            "Break research topic into important questions. ",
        ),
        (
            "human",
            "Create a research plan for this topic:\n\n{topic}",
        ),
    ]
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_version="v1",
    temperature = 0,
)

structured_model = model.with_structured_output(ResearchPlan)
chain = prompt | structured_model

def main():
    result = chain.invoke(
        {
            "topic": "How does speculative decoding improve LLM inference?"
        }
    )

    print(result)

if __name__ == "__main__":
    main()