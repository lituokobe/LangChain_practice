import os
from fastapi import FastAPI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import chat_agent_executor


os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"
#Tavily API Key is setup in the pycharm project settings, together with openai API key and langchain API key

model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')
#answer without agent
# result = model.invoke([HumanMessage(content = 'how\'s Singapore\'s weather today?')])
# print(result.content)

#Tavily is a search engine in LangChain, it needs another API
search = TavilySearchResults(max_results=2) #return max 2 results
#print(search.invoke('how\'s Singapore\'s weather today?'))

#bind the tool to the model
tools = [search]

#model_with_tools = model.bind_tools(tools)

#Model can automatically judge whether to use the tool to answer the question
#resp = model_with_tools.invoke([HumanMessage(content = 'What is 1 plus 1?')])
#
# print(f'Model_Result_Content: {resp.content}')
# print(f'Tools_Result_Content: {resp.tool_calls}')
#
# resp2 = model_with_tools.invoke([HumanMessage(content = 'how\'s Singapore\'s weather today?')])
#
# print(f'Model_Result_Content: {resp2.content}')
# print(f'Tools_Result_Content: {resp2.tool_calls}')

# create an agent
agent_executor = chat_agent_executor.create_tool_calling_executor (model, tools)

resp = agent_executor.invoke({'messages': [HumanMessage(content = 'What is 1 plus 1?')]})
print(resp['messages'])

resp2 = agent_executor.invoke({'messages': [HumanMessage(content = 'how\'s Singapore\'s weather today?')]})
print(resp2['messages'])
print(resp2['messages'][2].content)
print(resp2['messages'][3].content)
