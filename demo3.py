import os
from fastapi import FastAPI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langserve import add_routes

os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"

#practice to regulate the answers from AI in the given documents

model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

#prepare the data
documents = [
    Document(#Document is function in Langchain
        page_content="Dog is human's friend, and it's loyal and friendly.",
        metadata={"source": "mammal doc"},
    ),
    Document(
        page_content="Cat is independent, it prefers its own space",
        metadata={"source": "mammal doc"},
    ),
    Document(
        page_content="Fish is a pet for amateurs. It needs basic care",
        metadata={"source": "fish doc"},
    ),
    Document(
        page_content="Parrot is smart, it can imitate human speaking.",
        metadata={"source": "bird doc"},
    ),
    Document(
        page_content="Rabit is social, it needs more space to jump.",
        metadata={"source": "mammal doc"},
    ),
]

#create a vector space in langchain

vector_store = Chroma.from_documents(documents, embedding = OpenAIEmbeddings())

#search for similarity and find the score,
#it's more like a similarity distance score, the lower the score, the higher similarity
print(vector_store.similarity_search_with_score('coffee cat'))

#create a retriever
#retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)
#.bin(k=1) means to select the one with the highest similarity
retriever = vector_store.as_retriever()

print(retriever.batch(['coffe cat', 'shark']))

#make a prompt template
message = "use the context to answer the question: {question}, context: {context}"

prompt_temp = ChatPromptTemplate.from_messages([('human', message)])

#RunnablePassthrough allows up to pass the question from the user to prompt and model
chain = {'question':RunnablePassthrough(), 'context':retriever} | prompt_temp | model
resp = chain.invoke('What is a rabbit?')
print(resp.content)