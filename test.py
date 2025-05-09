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
from pydantic.v1 import BaseModel, Field
from yt_dlp import YoutubeDL
ydl_opts = {'quiet': True, 'extract_flat': True}


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"


# 聊天机器人案例
# 创建模型
#model = ChatOpenAI(model='gpt-4-turbo')
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

persist_dir = 'chroma_data_dir'  # 存放向量数据库的目录

# 一些YouTube的视频连接
urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
]

docs = []  # document的数组
# for url in urls:
#     # 一个Youtube的视频对应一个document
#     docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())


# with YoutubeDL(ydl_opts) as ydl:
#     info = ydl.extract_info("https://www.youtube.com/watch?v=HAn9vnJy6S4", download=False)
# print(info)
#
# print(len(docs))
# print(docs[0])
# # 给doc添加额外的元数据： 视频发布的年份
# for doc in docs:
#     doc.metadata['publish_year'] = int(
#         datetime.datetime.strptime(doc.metadata['publish_date'], '%Y-%m-%d %H:%M:%S').strftime('%Y'))
# #
# #
# print(docs[0].metadata)
# print(docs[0].page_content[:500])  # 第一个视频的字幕内容
#
# # 根据多个doc构建向量数据库
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=30)
# split_doc = text_splitter.split_documents(docs)
# # 向量数据库的持久化
# vectorstore = Chroma.from_documents(split_doc, embeddings, persist_directory=persist_dir)  # 并且把向量数据库持久化到磁盘