import os
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"

#create a model
model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14', temperature=0)
urls = ['https://americanliterature.com/author/louisa-may-alcott/book/little-women/part-one-chapter-one-playing-pilgrims',
        'https://americanliterature.com/author/louisa-may-alcott/book/little-women/part-one-chapter-two-a-merry-christmas',
        'https://americanliterature.com/author/louisa-may-alcott/book/little-women/part-one-chapter-three-the-laurence-boy',
        'https://americanliterature.com/author/louisa-may-alcott/book/little-women/part-one-chapter-four-burdens',
        'https://americanliterature.com/author/louisa-may-alcott/book/little-women/part-one-chapter-five-being-neighborly']

#Load the article
loader = WebBaseLoader(urls)
docs = loader.load() #have the full article

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

#Map-Reduce method to summarize content
#Phase 1: Map
map_template = """
{docs} is a set of documents. Please summarize the documents
"""
map_prompt =PromptTemplate.from_template(map_template)

map_llm_chain = LLMChain(llm=model, prompt=map_prompt)

#Phase 2: Reduce
reduce_template = """
{docs} os a set of summaries. Please summarize them to a final summary in Chinese, no more than 1000 characters.
"""
reduce_prompt =PromptTemplate.from_template(reduce_template)
reduce_llm_chain = LLMChain(llm=model, prompt=reduce_prompt)


#define a combined chain
combine_chain = StuffDocumentsChain(llm_chain=reduce_llm_chain, document_variable_name = 'docs')

reduce_chain = ReduceDocumentsChain(
    #this is the last chain
    combine_documents_chain = combine_chain,
    #chain in the middle
    collapse_documents_chain = combine_chain,
    #max token to group docs
    token_max = 4000
)

#connect all the chain
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain = map_llm_chain,
    reduce_documents_chain = reduce_chain,
    document_variable_name = 'docs',
    return_intermediate_steps=False # No need to return results in the middle. Default is False, no need to write this line.
)

result = map_reduce_chain.invoke(split_docs)
print(result['output_text'])


