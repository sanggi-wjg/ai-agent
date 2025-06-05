import uuid

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph_supervisor import create_supervisor

from agent.shared.tools import write_file_tool, web_scrap_tool


def create_file_writer_agent():
    llm = ChatOllama(model="qwen2.5:14b-instruct-q8_0")
    agent = create_react_agent(
        model=llm,
        tools=[write_file_tool],
        name="file_writer_agent",
        prompt="You are a writer. Your task is to write a file.",
        # debug=True,
    )
    return agent


def create_web_scrap_agent():
    llm = ChatOllama(model="qwen2.5:14b-instruct-q8_0")
    agent = create_react_agent(
        model=llm,
        tools=[web_scrap_tool],
        name="web_scrap_agent",
        prompt="You are a web scraper. Your task is to access the content of the given URL, summarize its key points, and create a new file containing the summarized information.",
        # debug=True,
    )
    return agent


thread_id = str(uuid.uuid4())
checkpointer = InMemorySaver()
store = InMemoryStore()

llm = ChatOllama(model="qwen2.5:14b-instruct-q8_0")
workflow = create_supervisor(
    [create_file_writer_agent(), create_web_scrap_agent()],
    model=llm,
    prompt="You are a supervisor managing a file_writer_agent and a web_scrap_agent.",
)
app = workflow.compile(
    # checkpointer=checkpointer,
    # store=store,
)

result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Summarize this https://github.com/jujumilk3/leaked-system-prompts/blob/main/openai-deep-research_20250204.md and write a file with the summary.",
            },
        ]
    },
    # config={"configurable": {"thread_id": thread_id}},
)
for m in result["messages"]:
    m.pretty_print()
