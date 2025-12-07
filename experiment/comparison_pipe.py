import os
import time
from typing import Any, Dict
os.environ["OPENAI_API_KEY"] = ""
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

def rag_retrieve(query, top_k: int = 3, filters: Dict[str, Any] | None = None):
    """Run KB retrieval via the ToolExecutor._exec_kb_retrieve handler.

    This uses the same executor path as the agent runtime so results
    are stored in an ExecutionState and logged similarly.
    Returns the list of hits (each a dict with 'text', 'source', etc.).
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from executor import ToolExecutor, ExecutionState

    execr = ToolExecutor()
    state = ExecutionState(slots={})
    args = {"query": query, "top_k": top_k, "filters": filters}
    start = time.time()
    success, error, elapsed = execr._exec_kb_retrieve(args, state, start)
    if not success:
        print(f"[rag_retrieve] kb_retrieve failed: {error}")
        return []
    return state.evidence.get("kb_hits", [])
@tool
def retrieve_data_from_rag(query: str) -> str:
    """Retrieve data from RAG system based on the query."""
    return rag_retrieve(query,3, None)


p = Path("experiment/RAG_SQL_LLM/data/parks_data.sqlite3").resolve()
print(f"Database path: {p}")
db = SQLDatabase.from_uri(f"sqlite:///{p}")
# Configure model
model = init_chat_model(
    "gpt-4.1-mini",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"]
)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools() + [retrieve_data_from_rag]

system_prompt = """
You are an agent designed to interact with a SQL database and a RAG system. Answer the user's question by querying the database or the RAG system or both as needed.
The rag system contains information about: 
1. Maintenance Activity Reference Guide - This document lists maintenance activity codes used by the Parks Department, including their descriptions, purposes, and typical maintenance frequencies.  Frequencies are expressed as **typical intervals under normal seasonal conditions** (April–October for most outdoor activities).
2. Sports Field Dimension Standards - This document provides standard dimensions for various sports fields and activities across different age groups and competition levels.
3. Mowing and Horticulture Maintenance Standards (South Zone) - This document summarizes the visual and operational criteria for mowing, trimming, and horticultural upkeep in the South Zone. It integrates performance metrics from both the Trim and Horticulture standards. All measurements follow local operational guidelines and apply to both the growing and non-growing seasons.

Use the following guidelines when interacting with the database:
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_agent(
    model=model,
    system_prompt=system_prompt,

    tools=tools)




# for step in agent.stream(
#     {"messages": [{"role": "user", "content": question}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()


questions1 = ["Compare Beaconsfield Diamond SW dimensions to Softball Men U11 standards. and calculate the differences if any.",
"Compare Beaconsfield Diamond SW dimensions to Softball Men U9, U7 standards. and calculate the differences if any.",
"Compare Bobolink Diamond EC dimensions to Softball Men U15 standards. and calculate the differences if any.",
"Compare Bobolink Diamond EC dimensions to Softball Men U13 standards. and calculate the differences if any.",
"Compare Bobolink Diamond EC dimensions to Softball Men U11 standards. and calculate the differences if any.",
"Compare Bobolink Diamond EC dimensions to Softball Men U9, U7 standards. and calculate the differences if any.",
"Compare Bobolink Diamond NE dimensions to Softball Men U15 standards. and calculate the differences if any.",
"Compare Bobolink Diamond NE dimensions to Softball Men U13 standards. and calculate the differences if any.",
"Compare Bobolink Diamond NE dimensions to Softball Men U11 standards. and calculate the differences if any.",
"Compare Bobolink Diamond NE dimensions to Softball Men U9, U7 standards. and calculate the differences if any.",
"Compare April 2025 mowing labor costs at Grade B fields with the recommended mowing frequency, and explain if the cost aligns with the standard.",
"Explain why mowing labor cost was highest in March 2025 based on mowing guidelines.",
"When was the last mowing at Cambridge park, is the frequency aligned with the standards?",
"Compare mowing cost across all parks in May 2025, and interpret based on the mowing standard.",
"Compare monthly mowing cost from April to June in 2025, are they aligned with mowing guidelines?",
"Which parks have the least mowing cost from April 2025 to June 2025? Are they aligned with mowing standards?",
"According to the policy, which parks are having maintenance activities due within 2 weeks?",
"Compare soccer U15 9v9 field length and width, are they aligned with field standards?",
"Which parks need Course Grooming in 10 weeks?",
"What is the cost of the activity in Stanley from February 2025 to March 2025? Is is aligned with the policy?",
"Per guidelines, what's the mowing frequency of Grade A (Soil)?",
"What is the frequency needed for paper picking?",
"What is the acceptable grass length range for Grade C (non-regulation, non-irrigated) fields during growing season?",
"Tell me the standard operating procedures for mowing",
"Show me baseball field requirements for U13",
"What's the pitching distance for female softball U17?",
"What is the frequency for invasive species removal?",
"Explain the equipment and cleanup steps required before trimming around Grade B fields.",
"What are the documented expectations for Goose Waste Removal, including when to escalate to special work orders?",
"Outline the horticulture weed-height limits and edging expectations cited in the standards document.",
"Which park had the highest total mowing labor cost in March 2025?",
"When was the last mowing at Cambridge Park?",
"Show mowing cost trend from January to June 2025",
"Compare mowing costs across all parks in March 2025",
"What is the mowing cost breakdown in March 2025?",
"Table only: compare soccer U12 9v9 field length and width",
"Rows only: U15 field length and width (diamonds), no explanation.",
"Which parks have the least mowing cost from June 2024 to May 2025?",
"What is the cost of the activity in Stanley from February 2025 to March 2025?",
"What is the latest activity in Stanley?",
"Table only: maintenance activities due within 3 weeks by park.",
"How many events occurred in June?",
"How many events occurred in May?",
"What is the activity code for paper picking?",
"What does activity code 290 represent?",
"What is the address of Balaclava Diamond SE field?",
"What is the Diamonds: Home to First Base Path - m attribute for Beaconsfield Diamond SW field?",
"How many events occurred at Angus Park?",
"How many events occurred at Cambridge Park?",
"How many parks are larger than 5 hectares?",
"How many parks are larger than 10 hectares?"]

import pandas as pd
results = []
for question in questions1:
    print(f"Question: {question}")
    start = time.time()
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
    except Exception as e:
        response = {"messages": [{"role": "error", "content": str(e)}]}
    elapsed = time.time() - start
    temp = {"question": question, "response": response["messages"][-1].content if not isinstance(response["messages"][-1], dict) else response["messages"][-1].get("content", ""), "time": elapsed}
    results.append(temp)
    time.sleep(10)
df = pd.DataFrame(results)
df.to_excel("experiment/benchmark.xlsx")