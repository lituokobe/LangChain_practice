import os
from fastapi import FastAPI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langserve import add_routes

os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"

#practice to keep record of the chat history and continue a conversation

#1. create the LLM
model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

#define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant, use {language} to reply everything you can.'),
    MessagesPlaceholder(variable_name='my_msg')
])


#4.get the chain
chain = prompt_template | model

#save the chat history
store = {} #save all users' chat history, key is sessionid

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


do_message = RunnableWithMessageHistory(
    runnable = chain,
    get_session_history = get_session_history,
    input_messages_key = 'my_msg' #send everytime when chatting
)

config = {'configurable': {'session_id': 'lt_123'}}

#first round
resp1 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content = 'Hello, this is Tuo.')],
        'language': 'English'
    },
    config = config
)

print(resp1.content)

#2nd round

resp2 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='Please say my name and great me in Spanish.')],
        'language': 'English'
    },
    config=config

)

print(resp2.content)

#3rd round, to return streaming data
for resp in do_message.stream({'my_msg': [HumanMessage(content='can you tell me a joke?')], 'language': 'Chinese'}, config = config):

    #every resp is a token
    print(resp.content, end = '-')
    pass