from typing import Union, Literal
from langchain_core.messages import HumanMessage

class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        # Optional legacy backend; local Qwen is injected into the original agents.
        from langchain_openai import ChatOpenAI, OpenAI
        # Determine model type from the kwargs
        model_name = kwargs.get('model_name', 'gpt-3.5-turbo') 
        if model_name.split('-')[0] == 'text':
            self.model = OpenAI(*args, **kwargs)
            self.model_type = 'completion'
        else:
            self.model = ChatOpenAI(*args, **kwargs)
            self.model_type = 'chat'
    
    def __call__(self, prompt: str):
        if self.model_type == 'completion':
            return self.model(prompt)
        else:
            return self.model(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            ).content
