import subprocess
import os
from hal_persona_prompt import prompt as HAL_PERSONA_PROMPT

# For OpenAI v1+ usage
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Ollama mode won't use this

def get_hal_system_message():
    return {"role": "system", "content": HAL_PERSONA_PROMPT}


class LLMClient:
    def __init__(self, backend, model_name, max_history=6, openai_api_key=None):
        self.backend = backend
        self.model_name = model_name
        self.max_history = max_history
        self.chat_history = []

        if backend == "openai":
            if OpenAI is None:
                raise ImportError("OpenAI package not found. Please install openai>=1.0.0")
            if not openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI backend")
            self.client = OpenAI(api_key=openai_api_key)

    def get_response(self, user_input):
        # Append user input with role 'user'
        self.chat_history.append({"role": "user", "content": user_input})

        # Trim chat history if it grows too long
        if len(self.chat_history) > self.max_history * 2:
            self.chat_history = self.chat_history[-self.max_history * 2 :]

        if self.backend == "openai":
            # models = self.client.models.list()
            # for model in models.data:
            #     print(model.id)

            system_message = get_hal_system_message()
            messages = [system_message] + self.chat_history

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=512,
                temperature=1,
            )
            reply = response.choices[0].message.content.strip()

            # Append assistant reply
            self.chat_history.append({"role": "assistant", "content": reply})
            return reply

        elif self.backend == "ollama":
            prompt = HAL_PERSONA_PROMPT + "\n" + "\n".join(
                f"{entry['role'].capitalize()}: {entry['content']}" for entry in self.chat_history
            ) + "\nHAL:"

            try:
                result = subprocess.run(
                    ["ollama", "run", self.model_name, prompt],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                reply = result.stdout.strip()
            except subprocess.CalledProcessError as e:
                reply = f"Error calling Ollama: {e}"

            # Append assistant reply
            self.chat_history.append({"role": "assistant", "content": reply})
            return reply

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
