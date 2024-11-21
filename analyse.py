import os
import re

import config
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.gigachat import GigaChat
#from yandex_chain import YandexLLM
from langchain_community.llms import YandexGPT
from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredURLLoader,
)

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, Tuple

import logging

from utils import transform_text_to_list, group_paragraphs
from prompts import (
    legal_prompt_agreement_type,
    
    legal_short_prompt,
    legal_warning
    )

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["LANGCHAIN_TRACING_V2"] = "true"

def get_loader(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".txt":
        return TextLoader(file_path, encoding='utf-8')
    elif extension == ".pdf":
        return PDFMinerLoader(file_path)
    elif extension in [".docx", ".doc"]:
        return UnstructuredWordDocumentLoader(file_path)
    elif extension in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(file_path)
    elif extension in [".pptx", ".ppt"]:
        return UnstructuredPowerPointLoader(file_path)
    elif extension in [".url", ".html", ".htm"]:
        return UnstructuredURLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
    

from typing import Optional
from pydantic import BaseModel, Field

# Initialize the assistant with the system prompt
llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY, model='gpt-4o-mini', temperature=0.1)

class Party(BaseModel):
    party_legal_name: Optional[str] = Field(..., description="Наименование стороны Договора")
    party_role: Optional[str] = Field(..., description="Наименование стороны в Договоре (например: именуемый в дальнейшем Заказчик)")

class Classification(BaseModel):
    document_type: str = Field(..., enum=["ДОГОВОР АРЕНДЫ", "ДОГОВОР ПОСТАВКИ", "ДОГОВОР ПОДРЯДА", "ДОГОВОР ОКАЗАНИЯ УСЛУГ", "АГЕНТСКИЙ ДОГОВОР", "ДРУГОЕ"])
    parties: Optional[List[Party]] = Field(..., description="Стороны Договора")


tagging_prompt = ChatPromptTemplate.from_template(legal_prompt_agreement_type)
tagging_chain = llm.with_structured_output(Classification)
classify_chain = tagging_prompt | tagging_chain

for file_path in os.listdir('./tests/'):
    try:
        loader = get_loader(f'./tests/{file_path}')
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])
        pattern = r'(?m)(?=^\d+(?:\.\d+)*\.\s)'
        #paragraphs  = re.split(pattern, full_text)
        paragraphs = transform_text_to_list(full_text)

        # Определяем тип документа
        zero_chunk = paragraphs[0]['text'] if paragraphs else full_text.strip()[:256]
        contract_metadata = classify_chain.invoke({"contract": zero_chunk})

        if contract_metadata.document_type != 'ДРУГОЕ':
            warnings = legal_warning[contract_metadata.document_type]
        else:
            raise Exception('Еще не реализовано.')

        for party in contract_metadata.parties:
            print(party.party_legal_name + " ИМЕНУЕМЫЙ " + party.party_role)

        legal_comments = []
        
        paras = group_paragraphs(paragraphs, 1)

        for paragraph in paras[1:]:
            human_prompt_template = "Верни комментарии для параграфа {paragraph} для стороны Договора {party}."
            system_message = SystemMessagePromptTemplate.from_template(legal_short_prompt)
            human_message = HumanMessagePromptTemplate.from_template(human_prompt_template)
            chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
            chain = chat_prompt | llm
            comments = chain.invoke({"legal_warnings": warnings, "contract": full_text, "paragraph": paragraph['text'], "party": contract_metadata.parties[0]})    
            legal_comments.append(f'Комментарий к параграфу {paragraph["text"]}:\n{comments.content}\n\n') #legal_comment = result.content

        with open(f'./results/{file_path}.txt', 'w', encoding='utf-8') as f:
            f.write("\n\n".join(legal_comments))
        print(f"{file_path} processed.")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

