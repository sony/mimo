from langchain_openai import AzureChatOpenAI

class LLM:

    def __init__(self, config):
        self.config = config
        self.openai_api_key = str(config["KEYS"][f"OPENAI_API_KEY"])
        self.azure_endpoint = str(config["KEYS"][f"OPENAI_ENDPOINT"])
        assert not "***********" in self.openai_api_key, "Error: Please input keys."
        self.loadLLM()


    def loadLLM(self):
        #self.llm = AzureChatOpenAI(deployment_name=self.config["LLM"]["deployment_name"], openai_api_version=self.config["LLM"]["openai_api_version"], openai_api_key = self.openai_api_key,\
        #    azure_endpoint=self.azure_endpoint, temperature=float(self.config["LLM"]["temperature"]))
        self.llm = AzureChatOpenAI(
                    deployment_name=str(self.config['KEYS']['gpt41_deployment_name']),
                    openai_api_version=str(self.config['KEYS']['gpt41_openai_api_version']),
                    openai_api_key = str(self.config['KEYS']['OPENAI_GPT4.1_API_KEY']),
                    azure_endpoint=str(self.config['KEYS']['OPENAI_GPT4.1_AZURE_OPENAI_ENDPOINT']),
                )
    
        