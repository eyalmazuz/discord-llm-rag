import argparse
import glob
import json
import os
import sqlite3

import bs4
import discord

from src.bot import LLMClient
from src.chats import OpenAIChat, GoogleChat
from src.document import Document
from src.encoder import OpenAIEncoder, SentenceTransformerInstructEncoder, GoogleEncoder
from src.faiss_store import FaissStore
from src.loaders import WebLoader, TextLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-size", type=int, default=1536, help="Size of the embedding to use"
    )
    parser.add_argument(
        "--knn-size", type=int, default=5, help="Amount of text to retrieve"
    )
    parser.add_argument(
        "--max-text-size",
        type=int,
        default=8192,
        help="maximum size of document to embed",
    )
    parser.add_argument(
        "--use-all-document",
        action="store_true",
        help="Whether to use the entire document as context or only the context",
    )
    parser.add_argument("--prompt-path", type=str, required=True)
    parser.add_argument(
        "--urls-path",
        type=str,
        help="path to a txt files containing URLs to crawl",
    )
    parser.add_argument(
        "--texts-path",
        type=str,
        help="glob-like pattern to where local text data is stored",
    )
    parser.add_argument("--database-path", type=str, help="Path to persisted database")

    return parser.parse_args()


def get_documents(
    database_path: str | None, urls_path: str | None, texts_path: str | None
) -> list[Document]:
    docs: list[Document] = []

    con = None
    if database_path is not None:
        con = sqlite3.connect(database_path)
        cur = con.cursor()
        results = cur.execute("SELECT text, metadata FROM web_pages").fetchall()
        docs += [
            Document(text=text, metadata=json.loads(metadata))
            for (text, metadata) in results
        ]

        print(f"Total documents after database loading: {len(docs)}")

    if urls_path is not None:
        with open(args.urls_path) as fd:
            urls = [url.strip() for url in fd.readlines()]
            if con:
                persisted_urls = [
                    result[0]
                    for result in cur.execute("SELECT url FROM web_pages").fetchall()
                ]
                urls = list(set(urls) - set(persisted_urls))

        print(f"Crawling {len(urls)} URLs")
        web_loader = WebLoader(
            urls=urls,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=(
                        "gh-content gh-canvas",
                        "article-header gh-canvas",
                        "page__main",
                    )
                )
            ),
        )
        web_docs = web_loader.load()
        if web_docs and con:
            print(f"Persisting {len(web_docs)} new documents")
            data = [
                (
                    document.metadata["source"],
                    document.text,
                    json.dumps(document.metadata),
                )
                for document in web_docs
            ]
            cur.executemany("INSERT INTO web_pages VALUES(?, ?, ?)", data)
            con.commit()

        docs += web_docs
        print(f"Total documents after crawling: {len(docs)}")

    if urls_path is not None:
        text_loader = TextLoader(glob.glob(args.texts_path))
        docs += text_loader.load()

        print(f"Total documents after loading local data: {len(docs)}")

    print()

    return docs


def main(args: argparse.Namespace) -> None:
    DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
    if DISCORD_TOKEN is None:
        raise ValueError("you must set the discord token as enviroment variable")

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)

    if GEMINI_API_KEY is None and OPENAI_API_KEY is None:
        raise ValueError(
            "you must set either openai api key or gemini key as enviroment variable"
        )

    with open(args.prompt_path) as fd:
        prompt = fd.read()

    docs = get_documents(args.database_path, args.urls_path, args.texts_path)
    # embedder = OpenAIEncoder(
    #     model="text-embedding-3-large",
    #     openai_key=OPENAI_API_KEY,
    #     embedding_size=args.embedding_size,
    # )
    # embedder = SentenceTransformerInstructEncoder(
    #     model="intfloat/multilingual-e5-large-instruct",
    #     embedidng_size=args.embedding_size,
    #     task_description="Given a question, retrieve relevant documents that best answer the question",
    # )
    embedder = GoogleEncoder(
        model="models/text-embedding-004",
        embedidng_size=args.embedding_size,
        api_key=GEMINI_API_KEY,
    )
    store = FaissStore(
        embedder, embd_size=args.embedding_size, max_text_size=args.max_text_size
    )
    store.add(docs, is_query=False)

    intents = discord.Intents.default()
    intents.message_content = True

    chat = OpenAIChat(prompt=prompt, openai_key=OPENAI_API_KEY)
    # chat = GoogleChat(prompt=prompt, google_key=GEMINI_API_KEY)

    client = LLMClient(
        intents=intents,
        store=store,
        chat=chat,
        knn_size=args.knn_size,
        use_all_document=args.use_all_document,
    )
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    args = parse_args()
    main(args)
