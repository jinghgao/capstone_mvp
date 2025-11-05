from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama
import os, pandas as pd
from sqlalchemy import create_engine
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, Any
from Data_layer import DataLayer
def sql_fall_back(query: str) -> Dict[str, Any]:
    """Function to execute SQL query with fall back to a different LLM if needed."""

    db = DataLayer().initialize_database()
    # print(db.show_tables_schemas())
    DATABASE_PATH = db.db_path
    engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=False)

    engine.dispose()

    # llm (unchanged)
    llm = ChatOllama(base_url="http://127.0.0.1:11434", model="llama3:8b", temperature=0, max_tokens=1000)

    # Connect LangChain to the file-backed SQLite via SQLAlchemy URI
    db = SQLDatabase.from_uri(f"sqlite:///{DATABASE_PATH}")
    table_schema = db.get_table_info(table_names=["event_data"])  # or build a string manually
    result = db.run("SELECT COUNT(*) FROM event_data WHERE date LIKE '2024-06-%';")
    # print("Events in June:", result)
    prompt = f"Schema: {table_schema}\nQuestion: How many events occurred in June? SQL queries return results as lists of tuples. For example, [(5860,)] means the result is 5860."

    # Create an agent that can generate SQL and run it against the database
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        handle_parsing_errors=True,
        agent_type="zero-shot-react-description",
        return_intermediate_steps=True
    )


    # Use the agent
    resp = agent_executor.invoke(query)
    #response = agent_executor.invoke("How many events occurred in May from the event data?")
    return {"answer_md": str(resp)}
