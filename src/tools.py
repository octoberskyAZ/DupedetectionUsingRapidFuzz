from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, trace, function_tool, OpenAIChatCompletionsModel, input_guardrail, GuardrailFunctionOutput
from typing import Dict
import os
from pydantic import BaseModel

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

instruction = ("You analyze the Orders data to identify duplicate")