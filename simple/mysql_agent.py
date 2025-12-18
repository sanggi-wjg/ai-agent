from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama

db = SQLDatabase.from_uri("mysql+pymysql://general_user:test1234@localhost:13307/dev")
llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)
agent = create_sql_agent(llm=llm, db=db, agent_type="tool-calling", verbose=True)

questions = [
    # "테이블 목록을 보여줘",
    # "유저 관련 테이블 알려줄래",
    "유저 중 오늘 접속한 유저들 조회하는 쿼리 알려줘",
    "지난 한달간 가장 많이 판매된 상품은 무엇인가요?",
    "지난 7일간 일별 신규 가입자 수를 보여줘",
]

for question in questions:
    response = agent.invoke({"input": question})
    print(response["output"])
    # for token in agent.stream({"input": question}):
    #     print(token, end="", flush=True)
