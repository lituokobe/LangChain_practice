import datetime
import os
from operator import itemgetter
from typing import Optional, List

import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_chroma import Chroma
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import chat_agent_executor
from pydantic import BaseModel, Field


os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"


model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')



