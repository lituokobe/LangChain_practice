import os
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import chat_agent_executor
import pymysql
pymysql.install_as_MySQLdb() #MySQLdb is not supported in this version of python, here is a substitute

os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"


model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

# sqlalchemy to connect database
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'world_bank_data'
USERNAME = 'root'
PASSWORD = ''
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db = SQLDatabase.from_uri(MYSQL_URI)

# Create a tp;;
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# User agent to integrate the database
system_prompt = """
You are an agent to communicate with SQL database.
You will be given a question, then you create a SQL query with correct grammar and execute it, review the result and return the answer.
Unless user specify the amount of results they want, limit the query results to less than 10.
You can sort the result, to return the the most relevant results from the SQL database.
You can use communication tools with the database. Before doing anything, you must check carefully.
If there is any mistakes in the execution, please re-write the SQL query and re-try.
Do you do any DML query to the data base, including insert, update, delete.

To start with, you should check the database' table, and see what can be queried.
Don't skip this step, and then search for information that is most relevant.
"""
system_message = SystemMessage(content=system_prompt)

# Create the agent
agent_executor = chat_agent_executor.create_tool_calling_executor(model = model, tools=tools, prompt = system_message)
#
resp = agent_executor.invoke({'messages': [HumanMessage(content='How many countries have gdp per capita higher than 10000?')]})
# resp = agent_executor.invoke({'messages': [HumanMessage(content='Which country has the longest name of capital?')]})
# resp = agent_executor.invoke({'messages': [HumanMessage(content='Which countries have life expectancy lower than 75?')]})

result = resp['messages']
print(result)
print(len(result))
# The last is the real answer
print(result[len(result)-1])