
from langchain_openai import AzureChatOpenAI

from pydantic import BaseModel, Field
import traceback

import ast


class TraditionalGPT():

    def __init__(self, config, key_type="GPT4", llm=None, output_desc=""):
        self.config = config
        self.llm = llm
        self.graph = None
        self.output_desc = output_desc
        self.key_type = key_type

    def loadLLM(self):
        self.llm = AzureChatOpenAI(deployment_name=self.config["LLM"]["deployment_name_" + self.key_type], openai_api_version=self.config["LLM"]["openai_api_version_" + self.key_type], openai_api_key = self.openai_api_key,\
            azure_endpoint = self.azure_endpoint, temperature =  float(self.config["LLM"]["temperature"]))
        
    def prompt(self, state, parse=True, config=None):
        class routeResponse(BaseModel):
            output: str = Field(description=self.output_desc)

        if self.llm is None:
            self.loadLLM()
        state = state["history"]["all"][0].content
        state = state.replace("{{", "{").replace("}}", "}")
        state += "\nMake sure to reply with only the dictionary."
        chain = self.llm
        try:
            result = chain.invoke(state)
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(state)
        print(result.content)
        if parse:
            return ast.literal_eval(result.content)
        else:
            return {"answer": result.content}