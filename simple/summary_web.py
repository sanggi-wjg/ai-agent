from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

loader = WebBaseLoader(
    web_path="https://www.lesswrong.com/posts/tqmQTezvXGFmfSe7f/how-much-are-llms-actually-boosting-real-world-programmer",
)
docs = loader.load()

set_debug(False)


def format_documents(documents):
    contents = "\n".join([doc.page_content.replace("\n\n", "\n") for doc in documents])
    return contents


system_template = """
You are a professional summarizer. Your task is to summarize the text while maintaining key details and clarity.
**No matter what language the input text is in, your response must always be in Korean. Even if the input text is in English, you must translate and summarize in Korean.**    

<REQUIREMENTS>  
When summarizing a text:  
1. Carefully analyze the entire content of the provided web page.  
2. Ensure that no critical information is omitted while keeping the summary concise.  
3. Adjust the length of the summary based on the original text:  
   - If the source is very long (e.g., 10+ A4 pages), summarize in approximately 1 A4 page.  
   - If the source is moderately long (e.g., 6 A4 pages), summarize in about half an A4 page.  
4. Structure the summary into two sections:  
   - **Topic:** A brief sentence summarizing the main theme.  
   - **Content:** Categorized key points organized logically.  
5. If the web page contains a specific objective, such as providing guidance, making an argument, or presenting research findings, ensure that this objective is explicitly mentioned.  
6. The summary should be self-contained and include enough context so that readers can understand it without referring to the original source.  
</REQUIREMENTS>

<OUTPUT_FORMAT>  
## 📌 주제  
[Briefly summarize the main idea]  

## 📖 내용  

## 🏷️ 주요 개념 & 핵심 정보  
- **[카테고리 1]**: Key points in a structured manner  
- **[카테고리 2]**: Additional details, if necessary.
- **[카테고리 3]**: Any other essential information. 

## 🎯 목적 / 핵심 메시지  
- Key message to deliver to main goal  

## ✅ 결론 & 활용 방법  
- Summary of how can leverage this information
</OUTPUT_FORMAT>
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": system_template},
        {"role": "user", "content": "Summarize the following content in KOREAN: {text}"},
    ]
)

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.1)
# llm = ChatOllama(model="exaone3.5:7.8b-instruct-fp16", temperature=0.1)

chain = prompt | llm
events = chain.stream({"text": format_documents(docs)})
for token in events:
    print(token.content, end="", flush=True)
