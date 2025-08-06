# Run:
# ollama pull llama3.2
# ollama pull nomic-embed-text
# ollama serve

import asyncio
import urllib.request
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector
import ollama


async def main():
    database_url = "postgresql://postgres:postgres@localhost:5432/postgres"
    conn = await asyncpg.connect(database_url)
    print("Connected to the database successfully!")

    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await register_vector(conn)

    await conn.execute("DROP TABLE IF EXISTS chunks")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id bigserial PRIMARY KEY,
            content text,
            embedding vector(768)
        )
    """)

    url = "https://raw.githubusercontent.com/pgvector/pgvector/refs/heads/master/README.md"
    dest = Path(__file__).parent / "README.md"
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    print(f"Downloaded {dest.name} successfully!")

    doc = dest.read_text()
    chunks = doc.split("\n## ")
    input = ["search_document: " + chunk for chunk in chunks]
    print(f"Split document into {len(input)} chunks.")

    embeddings = ollama.embed(model="nomic-embed-text", input=input).embeddings
    print(f"Generated embeddings for {len(embeddings)} chunks.")

    copy_status = await conn.copy_records_to_table(
        "chunks",
        records=zip(chunks, embeddings),
        columns=["content", "embedding"],
    )
    print(f"Copied {copy_status} rows to the database.")

    query = "What index types are supported?"
    input = f"search_query: {query}"
    query_embedding = ollama.embed(model="nomic-embed-text", input=input).embeddings[0]
    print("Generated query embedding.")

    result = await conn.fetch(
        """
        SELECT content, embedding <=> $1 AS distance
        FROM chunks
        ORDER BY distance
        LIMIT 5
        """,
        query_embedding,
    )
    context = "\n\n".join(row["content"] for row in result)

    prompt = f"Answer this question: {query}\n\n{context}"
    response = ollama.generate(model="llama3.2", prompt=prompt).response
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
