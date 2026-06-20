import os
from pathlib import Path

import chromadb
from google import genai
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_DB_PATH,
    TOP_K_RESULTS
)


class DocumentLoader:
    """
    Handles loading support documents from the data folder.
    Supports PDF, TXT, and Markdown files.
    """

    def __init__(self, data_path="data"):
        self.data_path = data_path


    def load_documents(self):
        documents = []

        for file_path in Path(self.data_path).glob("*"):

            extension = file_path.suffix.lower()

            try:
                if extension == ".pdf":
                    loader = PyPDFLoader(str(file_path))

                elif extension in [".txt", ".md"]:
                    loader = TextLoader(
                        str(file_path),
                        encoding="utf-8"
                    )

                else:
                    continue

                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = file_path.name

                documents.extend(docs)

            except Exception as error:
                print(f"Error loading {file_path}: {error}")

        return documents


    def split_documents(self, documents):
        """
        Split large documents into smaller chunks.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = splitter.split_documents(documents)

        return chunks


class VectorStore:
    """
    Handles Gemini embeddings and ChromaDB storage.
    """

    def __init__(self):
        load_dotenv()

        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH
        )

        self.collection = (
            self.chroma_client.get_or_create_collection(
                name="support_documents"
            )
        )


    def generate_embedding(self, text):
        """
        Convert text into vector embeddings.
        """

        response = self.client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
        )

        return response.embeddings[0].values


    def store_documents(self, chunks):
        """
        Store document chunks in ChromaDB.
        """

        for index, chunk in enumerate(chunks):

            embedding = self.generate_embedding(
                chunk.page_content
            )

            self.collection.add(
                ids=[f"chunk_{index}"],
                embeddings=[embedding],
                documents=[chunk.page_content],
                metadatas=[chunk.metadata]
            )


    def retrieve_documents(self, query, top_k=TOP_K_RESULTS):
        """
        Retrieve the most relevant document chunks.
        """

        query_embedding = self.generate_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_chunks = []

        if results["documents"]:

            for index in range(
                len(results["documents"][0])
            ):

                score = 1.0

                if (
                    "distances" in results
                    and results["distances"]
                ):
                    score = (
                        1 -
                        results["distances"][0][index]
                    )

                retrieved_chunks.append(
                    {
                        "text": results["documents"][0][index],
                        "source": (
                            results["metadatas"][0][index]
                            .get("source", "Unknown")
                        ),
                        "score": score
                    }
                )

        return retrieved_chunks