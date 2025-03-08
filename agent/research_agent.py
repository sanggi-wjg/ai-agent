from typing import Literal, TypedDict, Annotated, List, Dict

from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain.globals import set_debug
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command

from agent.shared.prompts import BASIC_AGENT_PROMPT
from agent.shared.tools import web_scrap_tool, write_file_tool, human_assistance_tool
from shared.utils import draw_graph_png

load_dotenv()
set_debug(False)


class State(TypedDict):
    user_query: Annotated[str, "User's query"]
    user_query_optimized: Annotated[str, "Optimized user's query"]
    # {'title': '6 Best Practices for Better Remote Performance Management', 'url': 'https://www.betterworks.com/magazine/better-performance-management-for-remote-teams/', 'content': 'These best practices provide a blueprint for improving remote performance outcomes and providing a better employee experience.', 'score': 0.7371019}
    search_response: Annotated[List[Dict[str, str]], "Search response"]
    summary_response: Annotated[List[str], "Output from research agent"]


def optimize_query_node(state: State) -> Command[Literal["search_node"]]:
    llm = ChatOllama(model="exaone3.5:7.8b", temperature=0.1)
    example_prompt = PromptTemplate.from_template("Question: {question}\nAnswer: {answer}")
    prompt = FewShotPromptTemplate(
        example_prompt=example_prompt,
        examples=[
            {
                "question": "k8s에 관해서 간략하게 요약해줄래?",
                "answer": "Kubernetes (K8s) overview and key concepts summary",
            },
            {
                "question": "집에서 할 수 있는 간단한 운동 알려줘",
                "answer": "Easy home workout routines",
            },
            {
                "question": "Easy home workout routines",
                "answer": "iPhone vs. Galaxy: Pros, cons, and best choice",
            },
            {
                "question": "효율적인 원격 근무 방법이 뭐야?",
                "answer": "Best practices for remote work productivity",
            },
            {
                "question": "AI 챗봇 개발하는 방법 알려줘",
                "answer": "How to build an AI chatbot: Tools and best practices",
            },
            {
                "question": "웹사이트 로딩 속도 빠르게 하는 방법이 뭐야?",
                "answer": "How to improve website loading speed: Optimization tips",
            },
            {
                "question": "사이버 보안이 중요한 이유가 뭐야?",
                "answer": "Why cybersecurity is crucial: Key threats and solutions",
            },
        ],
        suffix="Question: {input}",
        input_variables=["input"],
        prefix="Refine the following question for better web search accuracy. Use the examples below as a reference. Only return the optimized question without additional explanations or modifications.",
    )
    chain = prompt | llm | StrOutputParser()
    # chat_response = chain.invoke({"input": state["user_query"]})
    chat_response = "Best practices for remote work productivity"

    return Command(
        goto="search_node",
        update={"user_query_optimized": chat_response},
    )


def search_node(state: State) -> Command[Literal["summarize_node"]]:
    # search_tool = TavilySearchResults(max_results=3)
    # search_response = search_tool.invoke({"query": state["user_query_optimized"]})
    search_response = [
        {
            'title': '6 Best Practices for Better Remote Performance Management',
            'url': 'https://www.betterworks.com/magazine/better-performance-management-for-remote-teams/',
            'content': 'These best practices provide a blueprint for improving remote performance outcomes and providing a better employee experience.',
            'score': 0.7371019,
        },
        {
            'title': 'Best practices and lessons learned - NIH: Office of Human Resources',
            'url': 'https://hr.nih.gov/working-nih/workplace-flexibilities/remote-work/best-practices-and-lessons-learned',
            'content': 'Communication best practices\u200b\u200b Ensure conversations occur before the remote worker relocates to establish expectations and processes for team interaction.',
            'score': 0.6396691,
        },
        {
            'title': '10 Tips for Staying Productive When Working From Home',
            'url': 'https://www.travelers.com/resources/home/working-remotely/10-tips-for-staying-productive-when-working-from-home',
            'content': '1. Work out a schedule with your family · 2. Designate your own workspace · 3. Get up early – and dive right in · 4. Take breaks · 5. Eliminate the digital',
            'score': 0.6334335,
        },
    ]

    return Command(
        goto="summarize_node",
        update={"search_response": search_response},
    )


def summarize_node(state: State) -> Command[Literal[END]]:
    tools = [web_scrap_tool, write_file_tool, human_assistance_tool]

    llm = ChatOllama(model="qwen2.5:14b-instruct-q8_0", temperature=0.1)
    prompt = PromptTemplate.from_template(BASIC_AGENT_PROMPT)
    research_agent = AgentExecutor(
        agent=create_react_agent(llm, tools, prompt),
        tools=tools,
        max_iterations=5,
        verbose=True,
        handle_parsing_errors=True,
    )

    prompt = PromptTemplate.from_template(
        """
Your task is to access the content of the given URL, summarize its key points, and create a new file containing the summarized information.  

**Instructions:**  
1. Fetch the content from the following URL: {urls}  
2. Summarize the information in a clear and concise manner from the content.
3. Create a new file and store the summarized content in it.  

**Output Format:**  
- Title of the content  
- Key points (bullet points or numbered list)  
- A short paragraph summarizing the overall message  
- Ensure the summary is well-structured and easy to understand.

Finish the task once completed.
""".strip().format(
            urls=", ".join([sr["url"] for sr in state["search_response"]])
        )
    )
    chat_response = research_agent.invoke({"input": prompt})

    return Command(
        goto=END,
        update={"summary_response": chat_response},
    )


graph_builder = StateGraph(State)
graph_builder.add_node("optimize_query_node", optimize_query_node)
graph_builder.add_node("search_node", search_node)
graph_builder.add_node("summarize_node", summarize_node)

graph_builder.add_edge(START, "optimize_query_node")

graph = graph_builder.compile()
draw_graph_png("research_agent.png", graph)


stream = graph.stream({"user_query": "Docker를 사용하는 장점이 뭐야?"})
for s in stream:
    for agent, e in s.items():
        print(agent, e, flush=True)
