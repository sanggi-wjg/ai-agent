import uuid
from typing import Annotated

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.types import interrupt


@tool(
    name_or_callable="web_scrap_tool",
    description="Basic Web scrap tool, If you want to scrap text from url, you can use this tool.",
)
def web_scrap_tool(
    url: Annotated[str, "URL to scrap text from"],
) -> Annotated[str, "Text from URL"]:
    try:
        loader = WebBaseLoader(url)
        return loader.load()[0].page_content.replace("\n", "\t")

    except (SyntaxError, NameError) as e:
        return f"Sorry, I couldn't fetch the data from url({url}) cause by: {e}"


@tool(
    name_or_callable="write_file_tool",
    description="Basic Write file tool, If you want to write text to file, you can use this tool.",
)
def write_file_tool(
    # filepath: Annotated[str, "Path to write file"],
    text: Annotated[str, "Text to write to file"],
) -> Annotated[str, "File written successfully to {path}."]:
    filepath = f"{uuid.uuid4()}.md"
    try:
        with open(filepath, "w") as f:
            f.write(text)
        return f"File written successfully to {filepath}."

    except Exception as e:
        return f"Sorry, I couldn't write to file({filepath}) cause by: {e}"


@tool(
    name_or_callable="human_assistance_input_tool",
    description="When you requires human help or information, you can use this tool.",
)
def human_assistance_input_tool(
    query: Annotated[str, "Request to human"],
) -> Annotated[str, "Human response"]:
    return input(f"🤖 [Agent] 요청: {query}\n 💬 [User] 입력: ")


@tool(
    name_or_callable="human_assistance_interrupt_tool",
    description="When you requires human help or information, you can use this tool.",
)
def human_assistance_interrupt_tool(
    query: Annotated[str, "Request to human"],
) -> Annotated[str, "Human response"]:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


@tool(
    name_or_callable="python_tool",
    description="Use this to execute python code. If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.",
)
def python_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    try:
        repl = PythonREPL()
        result = repl.run(code)
        return (
            f"Successfully executed: {code}\nStdout: {result}\n\n"
            "If you have completed all tasks, respond with FINAL ANSWER."
        )
    except Exception as e:
        return f"Failed to execute. Error: {repr(e)}"
