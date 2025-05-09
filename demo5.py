import os
import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"


model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

#load the data from a blog
url_list = ['https://www.lituokobe.com/subjunctive-spanish-english-comparison-en1/']
loader = WebBaseLoader(
    web_path = url_list,
    bs_kwargs = dict(
        parse_only = bs4.SoupStrainer(class_=('blog-info', 'meta', 'blog-content')) #select the HTML classes on the blog page to decide what content to parse
    )
)
docs = loader.load()

print(docs)

#split the large text
#text = 'First of all, the most common situation is that when the principal clause of a Spanish sentence possesses certain subjective emotion, the subordinate clause that delivers the content of this emotion needs to be in the subjunctive mood. And this subjunctivity is not necessary in English.'
#splitter = RecursiveCharacterTextSplitter(chunk_size = 20, chunk_overlap = 4)
#every piece in the split has max 20 letters, no more overlapping of 4 letters with other lines
#it doesn't split a full word
#res = splitter.split_text(text)

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splits  = splitter.split_documents(docs)

#store
vectorstore = Chroma.from_documents(documents = splits, embedding = OpenAIEmbeddings())

#retriever
retriever = vectorstore.as_retriever()

#integration

#create a question template
system_prompt = """You are an assistant for Q&A tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say you don't know. Use max 3 sentences and keep the answer concise.\n
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        #MessagesPlaceholder("chat_history"),  #incldue the chat history
        ("human", "{context}"),
        #'context' is a must when using Document-Based Chains (create_stuff_documents_chain)
        # 'context' is a must when using RAG Pipelines, retrieving data from a vector store (retriever.as_retriever())
        ("human", "{input}"),
    ]
)

#create the chain
chain1 = create_stuff_documents_chain(model, prompt)
#chain2 = create_retrieval_chain(retriever, chain1)
#resp = chain2.invoke({'input': 'What is subjunctive mood in Spanish?', 'context': retriever.invoke('Please answer the question from this article only.')})
#print(resp['answer'])


"""
The retriever also need to understand the chat history by having a child chain.
"""

#create the child chain
context_system_prompt = '''
Using a chat history and the lastest user question which may be relevant in the chat hisotry, formulate a standalone question which can be understood. Without the chat history, do not answer the question, just reformulate it if needed, other wise return it as is.
'''

retriever_history_temp = ChatPromptTemplate.from_messages(
    [
        ("system", context_system_prompt),
        MessagesPlaceholder("chat_history"),  #incldue the chat history,
        ("human", "{context}"),
        ("human", "{input}"),
    ]
)

history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

#save the chat history
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#create a parent chain
chain = create_retrieval_chain(history_chain, chain1)

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history = get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    input_context_key='context',
    output_messages_key='answer',
)

#first round
resp1 = result_chain.invoke(
    {'input' : 'What is indicative mood?',
     'context': retriever.invoke('')},
    config = {'configurable': {'session_id' : 'lt1234'}}
)

print(resp1['answer'])

#second round
resp2 = result_chain.invoke(
    {'input' : 'Why is it special? Do not mention the other mood.' , #use 'it' to test the understanding of history
     'context': retriever.invoke('Use the info in the articl.')},
     config = {'configurable': {'session_id' : 'lt1234'}} # use the same session id, Ai will know 'it' refers to the concept mentioned before
     #config = {'configurable': {'session_id' : 'dff263'}} # use a different session id, Ai won't know  what is 'it'
)

print(resp2['answer'])