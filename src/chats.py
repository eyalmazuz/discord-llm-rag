from google import genai  # type: ignore
from openai import OpenAI


class OpenAIChat:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        prompt: str | None = None,
        openai_key: str | None = None,
    ) -> None:
        self.model = model
        if prompt is None:
            self.prompt = "Question: {question}\nContext:{context}\nAnswer:"
        else:
            self.prompt = prompt
        self.client = OpenAI(api_key=openai_key)

    def complete(self, message: str, context: str) -> str | None:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt.format(question=message, context=context),
                }
            ],
        )

        response = completion.choices[0].message.content
        return response


class GoogleChat:
    def __init__(
        self,
        model: str = "gemma-3-27b-it",
        prompt: str | None = None,
        google_key: str | None = None,
    ) -> None:
        self.model = model
        if prompt is None:
            self.prompt = "Question: {question}\nContext:{context}\nAnswer:"
        else:
            self.prompt = prompt
        self.client = genai.Client(api_key=google_key)

    def complete(self, message: str, context: str) -> str | None:
        completion = self.client.models.generate_content(
            model=self.model,
            contents=self.prompt.format(question=message, context=context),
        )

        response = completion.text
        return response
