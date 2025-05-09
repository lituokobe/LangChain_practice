import os
from operator import itemgetter
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pymysql
pymysql.install_as_MySQLdb() #MySQLdb is not supported in this version of python, here is a substitute

os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"

model = ChatOpenAI(model='gpt-3.5-turbo') #gpt-3.5-turbo will directly return SQL lines to execute, while other versions may not

#sqlalchemy is to connect with database
#set up the database info
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'world_bank_data'
USERNAME = 'root'
PASSWORD = ''
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db = SQLDatabase.from_uri(MYSQL_URI)

# Test if the connection is successful
#print(db.get_usable_table_names())
#print(db.run("SELECT * FROM `world-politics_2`;"))

# integrate the model and database info, only generate SQL based on my question
test_chain = create_sql_query_chain(model, db)
# resp = test_chain.invoke({'question': 'What is the GDP per capta of Spain?'})
# print(resp)
# The line above will generate the SQL line to find the aswner.

answer_prompt = PromptTemplate.from_template(
    """Provide the user question, SQL query and SQL result as follows, answer user question.
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    answer: """
)
# Create a tool to execute SOL query
execute_sql_tool = QuerySQLDataBaseTool(db=db)

chain = (RunnablePassthrough.assign(query=test_chain).assign(result=itemgetter('query') | execute_sql_tool)
        #test_chain use the model and the data base to generate sql query, execute_sql_tool executes the query
         | answer_prompt
         | model
         #The model here didn't do too much in finding the information, it just combine the results from the SQL work
         | StrOutputParser()
         )

rep = chain.invoke(input={'question': 'What is the gap of gdp per capita between Cameroon and China?'})
print(rep)