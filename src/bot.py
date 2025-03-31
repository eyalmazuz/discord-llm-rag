import discord
import numpy as np

from .document import Document


class LLMClient(discord.Client):
    def __init__(
        self,
        *,
        intents: discord.Intents,
        store,
        chat,
        knn_size,
        use_all_document,
    ):
        super().__init__(intents=intents)
        self.threads: dict[int, discord.Thread] = {}

        self.chat = chat
        self.store = store
        self.knn_size = knn_size
        self.use_all_document = use_all_document

    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("------")

    async def on_message(self, message):
        # Ignore messages from the bot itself
        if message.author == self.user:
            return

        if message.channel.id not in [] and message.channel.id not in self.threads:
            return

        # If the message is sent in a normal channel (not a thread)
        if not isinstance(message.channel, discord.Thread):
            documents, scores = self.get_context(message.content)

            sources = "\n".join(
                [
                    f"{url}: {score:.3f}"
                    for (url, score) in zip(
                        [document.metadata["source"] for document in documents], scores
                    )
                ]
            )

            if self.use_all_document:
                context = "\n".join(
                    [
                        document.metadata["original-text"]
                        if "original-text" in document.metadata
                        else document.text
                        for document in documents
                    ]
                )
            else:
                context = "\n".join([document.text for document in documents])

            response = self.chat.complete(message.content, context)

            answer = f"{response}\nSources:\n{sources}"

            # Add a red question mark reaction to the original message
            await message.add_reaction("❓")
            # Create a thread attached to the original message
            thread = await message.create_thread(name="Answer here")

            self.threads[thread.id] = message
            # Send an instruction message in the thread (optional)
            await thread.send("""
            "Bip bap bop, I'm a bot, I might get it wrong, so trust me not.
            Always check your sources, stay alert, Or wait for a human to assert.
            But here’s my answer, take it, see— I hope it helps, from me to thee!"
            """)

            await thread.send(answer)

        else:
            # If the message is in a thread, check for the ?solved command
            if message.content.strip() == "?solved":
                thread = message.channel
                # Look up the original message using our stored mapping
                original_message = self.threads.get(thread.id)
                if original_message is None:
                    # Could not determine the original message
                    return

                # Verify that the person issuing ?solved is the original message author
                if original_message.author.id != message.author.id:
                    # Optionally inform the user they aren't allowed to solve this thread
                    return

                # Archive (close) the thread
                await thread.edit(archived=True)
                # Remove the red question mark reaction from the original message
                await original_message.remove_reaction("❓", self.user)
                # Add a green check mark reaction to indicate the question is solved
                await original_message.add_reaction("✅")
                # Remove the entry from the mapping, as it's no longer needed
                del self.threads[thread.id]

    def get_context(self, message: str) -> tuple[list[Document], np.ndarray]:
        text = message.replace("\n", " ")
        documents, scores = self.store.get(text, k=self.knn_size, is_query=True)

        return documents, scores
