import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain

os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"

#create a model
model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14', temperature=0)

#Load the article
loader = WebBaseLoader('https://www.lituokobe.com/fight-against-misinformation-of-covid-19/')
docs = loader.load() #have the full article

# Directly use Stuff method to summarize text
#chain = load_summarize_chain(model, chain_type='stuff')

#include prompts in the Stuff method
prompt_template = """
Write a summary in Chinese based on the following content: {context}. Please limit to 200 characters."""
prompt = PromptTemplate.from_template(prompt_template)

# chain = LLMChain(llm=model, prompt=prompt)
# stuff_chain = StuffDocumentsChain(llm_chain=chain, document_variable_name='context')
# In the 2 lines of code above, LLMChain and StuffDocumentsChain are deprecated, so use the code below (as of May 2025).

stuff_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

result = stuff_chain.invoke({"context": docs})
print(result)