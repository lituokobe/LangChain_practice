import os

from fastapi import FastAPI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"


#1. create the LLM
model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

#2. prepare the prompt
msg = [
    SystemMessage(content = "Please translate the following content to Spanish"),
    HumanMessage(content = "Hello, how's the weather today?")
]

#result = model.invoke(msg)
#print(result)

#3.parse the data
parser = StrOutputParser()
#print(parser.invoke(result))

#define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'Please translate the following content to {language}.'),
    ('user', '{text}')
])

#4.get the chain
#chain = model | parser
chain = prompt_template | model | parser

#5. user the chain
#print(chain.invoke(msg))
print(chain.invoke({'language': 'Japanese', 'text': 'Do you want to have a massage?'}))

#deploy the program to a service

app = FastAPI(title = 'Language Translation Service', version = '1.0', description='Use LangChain to translate.')

add_routes(
    app,
    chain,
    path = '/chainDemo'
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)

#Visit the service on this Mac at http://localhost:8000/chainDemo/playground/
