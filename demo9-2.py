import os
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.synthetic_data import create_data_generation_chain
from pydantic import BaseModel, Field


os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "pr-shadowy-maybe-22"
os.environ['LANGCHAIN_PROJECT'] = "Tuo-Demo"

#create a model that can generate content
model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14', temperature=0.8)

#create a chain
chain = create_data_generation_chain(model)

#create structured data,
#below is the medical bill example, this one is so American
class MedicalBilling(BaseModel):
    patient_id: int
    patient_name: str
    diagnosis_code: str
    procedure_code: int
    total_charge: float
    insurance_claim_amount: float

#some sample input

examples = [
    {
        "example": "Patient ID: 124786, Patient Name: Benjamin Maharaja, Diagnosis Code: L70.9, Procedure Code: 99356, Total Charge: $700, Insurance Claim Amount: $540.2"
    },
    {
        "example": "Patient ID: 789192, Patient Name: Xingpeng Rivotali, Diagnosis Code: K34.5, Procedure Code: 12943, Total Charge: $260, Insurance Claim Amount: $68.94"
    },
    {
        "example": "Patient ID: 275678, Patient Name: Mitsutoyo Kyvrovski, Diagnosis Code: A21.9, Procedure Code: 93614, Total Charge: $2760.45, Insurance Claim Amount: $1139"
    },
]

#create a prompt template to guide AI to generate structured data as stipulated
openai_template = PromptTemplate(input_variables = ['example'],
                                 template = "{example}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    examples=examples,
    example_prompt=openai_template,
    input_variables=['subject', 'extra']
)

#create a generator for structured data
generator = create_openai_data_generator(
    output_schema=MedicalBilling,
    llm=model,
    prompt = prompt_template,
)

# use the generator
result = generator.generate(
    subject='Medical Billing',
    extra='Use rare names that combines different cultures, total charge should be normally distributed and the minimum amount is $1000',
    runs = 15 #numbers of data generated
)

print(result)

