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
persist_dir = 'chroma_data_dir'
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

#The Youtube transcript tool is outdated, we will use a tool to get content from webpage.

urls = [
    'https://www.lituokobe.com/subjunctive-spanish-english-comparison-en2/'
]

# loader = WebBaseLoader(
#     web_path = urls,
#     bs_kwargs = dict(
#         parse_only = bs4.SoupStrainer(class_=('blog-info', 'meta', 'blog-content')) #select the HTML classes on the blog page to decide what content to parse
#     )
# )
#
# docs = loader.load()
#
# print(len(docs))
# print(docs[0])
#
#
# #split the doc
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=30)
# split_doc = text_splitter.split_documents(docs)
#
# #make the vector data persistent by saving it
# vectorstore = Chroma.from_documents(split_doc, embeddings, persist_directory=persist_dir)

#after saving it, let's try load the vector data
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

#test similarity
# result = vectorstore.similarity_search_with_score('This is an example of subjunctive mood in Spanish.')
# print(result[0][0].metadata)

system = """You are an expert at Spanish and English grammar. \
You have access to a database of tutorial of subjunctive mood in Spanish and English. \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
#pydanitc
class Search(BaseModel):
    """
    Define a data model
    """
    #based on content similarity
    query: str = Field(None, description="Similarity search query applied to the content.")
    #You can also search based on query on the year, if the information is available in the content
    #publish_year: Optional[int] = Field(None, description='Year of the grammar example')

#The chain below will craft the question based on the Search class, and parse query to a proper data structure, but without answer.
chain = {'question': RunnablePassthrough()} | prompt | model.with_structured_output(Search)

resp1 = chain.invoke('Can you give somes example of variation of inversion in the Unrealistic Scenario in English?')
print(resp1)

def retrieval(search: Search) -> List[Document]:
    _filter = None
    #if search.publish_year:
        # make a qeury conditin based on publish year
        # "$eq" is fixed grammar in Chroma vector
        #_filter = {'publish_year': {"$eq": search.publish_year}}
        #This will give us not only the similar content, but also meet some conditions, e.g. publish year.
        #Though it is not useful in this practice
    return vectorstore.similarity_search(search.query, filter=_filter)

new_chain = chain | retrieval

result = new_chain.invoke('Can you give some example of variation of inversion in the Unrealistic Scenario in English?')
print(result[1].page_content)
