import os

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"

#create a model to classify sentiments
model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14', temperature=0) # we don't need any randomness for classification, make the temperature zero

#create structured data,
# class Classification(BaseModel):
#     sentiment: str = Field(description="Sentiment of the text.")
#     aggressiveness: int = Field(description="Aggressiveness of the text. The higher the more aggressive. minimum is 0, maximum is 10.")
#     language: str = Field(description="Language of the text.")

class Classification(BaseModel):
    sentiment: str = Field(..., enum=['happy', 'neutral', 'angry', 'sad', ], description="Sentiment of the text.")
    aggressiveness: int = Field(..., enum=[0,1,2,3,4,5,6,7,8,9,10], description="Aggressiveness of the text. The higher the more aggressive. minimum is 0, maximum is 10.")
    language: str = Field(description="Language of the text.")

tagging_prompt = ChatPromptTemplate.from_template(
    """
    Extract information from the following text, only extract the attributes in 'Classification'.
    Text:
    {input}
    """)

chain = tagging_prompt | model.with_structured_output(Classification)

input_text = '''China on Friday (May 9) said sales to the United States slumped last month while its
total exports topped forecasts, as Beijing fought a gruelling trade war with its superpower rival.
Trade between the world's two largest economies has nearly skidded to a halt since US President Donald
Trump imposed various rounds of levies on China that began as retaliation for Beijing's alleged role
in a devastating fentanyl crisis.'''

input_text2 = 'آن مدیر ارشدی که در خدمت است، یک عوضی کثیف است که دیک سوسک را می مکد.'

input_text3 = 'Me alegro mucho de verte. Ha pasado mucho tiempo desde la última vez que te vimos.'

result1: Classification = chain.invoke({'input': input_text})
print(result1)

result2 = chain.invoke({'input': input_text2})
print(result2)

result3 = chain.invoke({'input': input_text3})
print(result3)


