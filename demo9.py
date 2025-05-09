import datetime
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.synthetic_data import create_data_generation_chain


os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"

#create a model that can generate content
model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14', temperature=0.8)

#create a chain
chain = create_data_generation_chain(model)

#create some random data

# result =chain.invoke({ #some key words to generate random text
#     'fields' : ['blue', 'yellow'],
#     'preferences' : {}
# })

result =chain.invoke({ #some key words to generate random text
    'fields' : {'color': ['yellow', 'grey'], 'element': ['fat', 'snake']},
    'preferences' : {'style':'make it like a poem'}
})

print(result)


