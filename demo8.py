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
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"


model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

#pydantic: process data, validate data, define data format, virtualize or devirtualize data, convert data type.
#pydantic makes OOP more convenient.
class Person(BaseModel):
    """
    a data module about a person
    """
    name: Optional[str] = Field(default=None, description='this is the person name')

    hair_color: Optional[str] = Field(
        default=None, description="the hair color of the person"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="height of the person by meters"
    )


class ManyPerson(BaseModel):
    """
    data module about many persons
    """
    people: List[Person]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert on extracting algorithm, and only extract relevant information from unstructured content. If you don't know the value of the attribute, return null for the attribute value.",
        ),
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

#the output is structured. AI helps to extract the structured data from the input text
chain = {'text': RunnablePassthrough()} | prompt | model.with_structured_output(schema=ManyPerson)
text = 'There is a girl coming by, with long and brown hair, about 165 cm tall. Her godfather Diego is next to her and 20cm taller than her. Diego has a lighter hair color than his god-daughter.'
resp = chain.invoke(text)
print(resp)


