import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

search_tool = SerperDevTool()
from langchain_openai import ChatOpenAI

researcher = Agent(
    role = "Senior Researcher", 
    goal = "Look up the latest news in AI for software development",
    backstory = """You work at a leading tech think tank.
    Your expertise lies in searching Google for AI software development tools.
    You have a knack for dissecting complex data and presenting it in a digestible format.""",
    verbose = False,
    allow_delegation = False,
    tools = [search_tool],
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
)

writer = Agent(
    role = "Professional Short-Form Writer", 
    goal = "Summarize the latest advancements in AI software development news in a concise article",
    backstory = """You are a renowned Content Strategist, known for your insightful and engaging articles.
    You transform complex concepts into compelling narratives.""",
    verbose = True,
    allow_delegation = True,
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
)

task1 = Task(
    description = "Research the latest advancements in AI software development",
    expected_output = "A summary of the latest advancements in AI software development",
    agent = researcher
)

task2 = Task(
    description = "Write a concise article on the latest advancements in AI software development",
    expected_output = "A well-written article on the latest advancements in AI software development",
    agent = writer
)

crew = Crew(
    agents = [researcher, writer],
    tasks = [task1, task2],
    verbose = 1
)

results = crew.kickoff()

print("#########################################")
print(results)

