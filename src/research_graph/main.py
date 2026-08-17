import os 
from dotenv import load_dotenv
from pydantic import BaseModel,Field 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.runnables import RunnableLambda

load_dotenv()

class ResearchPlan(BaseModel):
    topic:str = Field(description = "The research topic")
    questions: list[str] = Field(
        description="Important questions that should be investigated"
    )

class ResearchBrief(BaseModel):
    title:str= Field(
        description="A concise title for the research"
    )

    summary:str = Field(
        description="A short explanation of what should be investigated"
    )

    questions:list[str] = Field(
        description = "The key research questions"
    )

planner_prompt = ChatPromptTemplate.from_messages(
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

brief_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research editor."
            "Turn a research plan into a concise research brief",
        ),
        (
            "human",
            "create a research brief from this research plan:\n\n{plan}",
        ),
    ]
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
)

structured_model = model.with_structured_output(
    schema=ResearchPlan.model_json_schema(),
    method="json_schema",
)

brief_model = model.with_structured_output(
    schema=ResearchBrief.model_json_schema(),
    method="json_schema",
)

planner_chain = planner_prompt | structured_model
brief_chain = brief_prompt | brief_model

def prepare_plan(plan:ResearchPlan):
    return{
        "plan":plan
    }

prepare_plan_runnable = RunnableLambda(prepare_plan)

research_chain = (
    planner_prompt
    | structured_model
    | prepare_plan_runnable
    | brief_prompt
    | brief_model
)
def main():
    result = research_chain.invoke(
        {
            "topic": "How does speculative decoding improve LLM inference?"
        }
    )
 
    
    

    print(result)
    print(type(result))

if __name__ == "__main__":
    main()