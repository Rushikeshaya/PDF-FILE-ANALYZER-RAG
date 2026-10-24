import os

import psycopg
from pgvector.psycopg import register_vector
from pgvector import Vector
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:ragpassword@localhost:5433/offline_rag"
)


def get_connection():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def initialize_database():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL
                        REFERENCES documents(id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding VECTOR(384) NOT NULL,
                    search_vector TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', chunk_text)
                    ) STORED
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_search_idx
                ON document_chunks USING GIN(search_vector);
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON document_chunks USING hnsw
                (embedding vector_cosine_ops);
            """)


def document_exists(file_hash):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM documents WHERE file_hash = %s",
                (file_hash,)
            )
            return cur.fetchone() is not None


def save_document(filename, file_hash, chunks, embeddings):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, file_hash)
                VALUES (%s, %s)
                ON CONFLICT (file_hash) DO NOTHING
                RETURNING id
                """,
                (filename, file_hash)
            )

            row = cur.fetchone()

            if row is None:
                return False

            document_id = row[0]

            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO document_chunks
                    (document_id, page_number, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        chunk["page_number"],
                        chunk["text"],
                        Vector(embedding.tolist())
                    )
                )

    return True


def hybrid_search(question, query_embedding, limit=5):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH semantic_results AS (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            ORDER BY embedding <=> %s::vector
                        ) AS rank
                    FROM document_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT 20
                ),
                keyword_results AS (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            ORDER BY ts_rank(
                                search_vector,
                                websearch_to_tsquery('english', %s)
                            ) DESC
                        ) AS rank
                    FROM document_chunks
                    WHERE search_vector @@
                        websearch_to_tsquery('english', %s)
                    ORDER BY ts_rank(
                        search_vector,
                        websearch_to_tsquery('english', %s)
                    ) DESC
                    LIMIT 20
                ),
                combined_results AS (
                    SELECT id, SUM(score) AS combined_score
                    FROM (
                        SELECT id, 1.0 / (60 + rank) AS score
                        FROM semantic_results

                        UNION ALL

                        SELECT id, 1.0 / (60 + rank) AS score
                        FROM keyword_results
                    ) AS rankings
                    GROUP BY id
                )
                SELECT
                    dc.chunk_text,
                    dc.page_number,
                    d.filename,
                    combined_results.combined_score
                FROM combined_results
                JOIN document_chunks dc
                    ON dc.id = combined_results.id
                JOIN documents d
                    ON d.id = dc.document_id
                ORDER BY combined_results.combined_score DESC
                LIMIT %s
                """,
                (
                    Vector(query_embedding.tolist()),
                    Vector(query_embedding.tolist()),
                    question,
                    question,
                    question,
                    limit
                )
            )

            rows = cur.fetchall()

    return [
        {
            "text": row[0],
            "page_number": row[1],
            "filename": row[2],
            "score": float(row[3])
        }
        for row in rows
    ]