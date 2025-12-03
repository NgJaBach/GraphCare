import os
import openai

with open("../../resources/openai.key", 'r') as f:
    key = f.readlines()[0][:-1]

class ChatGPT:
    def __init__(self):
        # Setting the API key to use the OpenAI API
        openai.api_key = key
        self.messages = []

    def chat(self, message):
        while True:
            try:
                self.messages.append({"role": "user", "content": message})
                response = openai.ChatCompletion.create(
                    model="gpt-5-nano",
                    messages=self.messages
                )
                # self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
                return response["choices"][0]["message"]
            except:
                continue