BASIC_AGENT_PROMPT = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""".strip()

QUERY_WRITER_INSTRUCTIONS = """
Your goal is to generate a targeted web search query. The query will gather information related to a specific topic.
Always respond in Korean.

<TOPIC>
{topic}
</TOPIC>""".strip()

SUMMARIZE_INSTRUCTIONS = """
Generate a high-quality summary of the web search results and keep it concise / related to the user topic.
Always respond in Korean.

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully.                                                    
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user topic, skip it.                                                            
4. Ensure all additions are relevant to the user's topic.                                                         
5. Verify that your final output differs from the input summary.                                                                                                                                                            
</REQUIREMENTS>

<FORMATTING>
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.  
</FORMATTING>""".strip()

REFLECTION_INSTRUCTIONS = """
You are an expert research assistant analyzing a summary about {topic}.
Always respond in Korean.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
</GOAL>

<REQUIREMENTS>
Ensure the follow-up question is self-contained and includes necessary context for web search.
</REQUIREMENTS>

Provide your analysis:""".strip()

API_TEST_PLAN_INSTRUCTIONS = """
You are an API test planner. Your task is to generate API request test plans based on the given API specification.

<REQUIREMENTS>
When planning an API request:
1. Carefully analyze the provided API specification.
2. Generate a structured test request plan that is valid and can be executed successfully.
3. Ensure the plan follows the API constraints (e.g., required parameters, request body format).
4. Consider multiple test scenarios, including normal cases, edge cases, and error cases.
5. Output the test plan in the following JSON format:
</REQUIREMENTS>

<API_SPEC>
{api_spec}
</API_SPEC> 
""".strip()
