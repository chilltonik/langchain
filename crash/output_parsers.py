import os

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
from langchain_core.output_parsers import (CommaSeparatedListOutputParser,
                                           JsonOutputParser, ListOutputParser,
                                           StrOutputParser)
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=float(os.getenv("LLM_TEMPERATURE")),
    max_tokens=os.getenv("LLM_MAX_TOKENS"),
    verbose=True,
)


def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            {"role": "system", "content": "Tell me a joke about the following subject"},
            {"role": "user", "content": "{input}"},
        ]
    )

    parser = StrOutputParser()
    chain = prompt | model | parser

    return chain.invoke({"input": "cat"})


# response = call_string_output_parser()
# print(response)


def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            {
                "role": "system",
                "content": "Give me 10 synonymous to the next word. Return the response as a comma separated list.",
            },
            {"role": "user", "content": "{input}"},
        ]
    )

    parser = CommaSeparatedListOutputParser()
    chain = prompt | model | parser

    return chain.invoke({"input": "sadness"})


# response = call_list_output_parser()
# print(response)


def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            {
                "role": "system",
                "content": "Instruct information from the following phrase.\n Formatting instructions: {format_instructions}",
            },
            {"role": "user", "content": "{phrase}"},
        ]
    )

    class Person(BaseModel):
        name: str = Field(description="The person's name")
        gender: str = Field(description="The person's gender")
        age: int = Field(description="The person's age")

    parser = JsonOutputParser(pydantic_object=Person)
    chain = prompt | model | parser

    return chain.invoke(
        {
            "phrase": "Anton is 30 years old",
            "format_instructions": parser.get_format_instructions(),
        }
    )


response = call_json_output_parser()
print(response)
