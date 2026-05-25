import hashlib
from pathlib import Path

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv


# =============================================================================
# SIMPLE RAG DEMO
# =============================================================================
# RAG means Retrieval-Augmented Generation.
#
# The idea:
#   1. Start with some text documents.
#   2. Turn the text into vectors using an embedding model.
#   3. Store those vectors in a vector database, here FAISS.
#   4. For a question, retrieve the most relevant documents.
#   5. Give the retrieved text to an LLM so it can answer from that context.


# Keep paths relative to this script so it works from any terminal folder.
PROJECT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = Path(__file__).name
EMBEDDINGS_DIR = PROJECT_DIR / "embeddings" / "simple_rag"

# Models
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHAT_MODEL_NAME = "gpt-4o-mini"


# 0. Load the OpenAI API key from rag/.env
load_dotenv(PROJECT_DIR / ".env")


# 1. Example text documents
documents = [
    "Zürich is Switzerland’s largest city, but it is not the capital.",
    "Zürich began as a Roman customs station called Turicum near today’s Lindenhof area.",
    "Lake Zürich flows into the Limmat River, which runs through the city’s historic old town.",
    "Switzerland does not formally define a capital, it has a federal city.",
    "Bern is the federal city and seat of the Swiss federal government.",
]


# 2. Turn text into vectors using an embedding model
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)


def cache_key_for_docs(docs):
    """Create a stable cache key for these documents and this embedding model."""
    # If the documents or embedding model change, the key changes too.
    # That gives us a fresh FAISS cache instead of reusing old embeddings.
    cache_text = EMBEDDING_MODEL_NAME + "\n" + "\n".join(docs)
    return hashlib.sha256(cache_text.encode("utf-8")).hexdigest()[:12]


def save_cache_metadata(cache_path, docs):
    """Write a small note so we know which script created this FAISS index."""
    metadata = (
        f"script={SCRIPT_NAME}\n"
        f"embedding_model={EMBEDDING_MODEL_NAME}\n"
        f"document_count={len(docs)}\n"
    )
    (cache_path / "metadata.txt").write_text(metadata, encoding="utf-8")


# 3. Store the vectors in FAISS, our vector database
# The first run creates embeddings and saves a FAISS index.
# Later runs load the saved index, so we do not pay to embed the same text again.
EMBEDDINGS_DIR.mkdir(exist_ok=True)
vectorstore_path = EMBEDDINGS_DIR / f"faiss_{cache_key_for_docs(documents)}"

if vectorstore_path.exists():
    print(f"Loading FAISS index from {vectorstore_path}")
    vectorstore = FAISS.load_local(
        folder_path=str(vectorstore_path),
        embeddings=embedding_model,
        # This is safe here because we only load a local file created by this script.
        allow_dangerous_deserialization=True,
    )
else:
    print("Creating FAISS index and saving embeddings...")
    # FAISS.from_texts embeds the documents and stores the vectors in FAISS.
    vectorstore = FAISS.from_texts(
        texts=documents,
        embedding=embedding_model,
    )
    vectorstore.save_local(str(vectorstore_path))
    save_cache_metadata(vectorstore_path, documents)
    print(f"Saved FAISS index to {vectorstore_path}")


# 4. Create a retriever.
# A retriever searches FAISS and returns the most relevant documents.
# k=2 means: return the top 2 matching documents for each question.
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# 5. Define the RAG prompt
# The model should only answer from retrieved context, not from general knowledge.
template = """
Answer the question based only on the following context:
If the answer is not in the context, say:
"I don't know based on the provided context."

---
Context:
{context}
---

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)


# 6. Define the chat model
model = ChatOpenAI(model=CHAT_MODEL_NAME, temperature=0)


# 7. Build the answer chain.
# This part only handles:
#   context + question -> prompt -> model -> string answer
answer_chain = prompt | model | StrOutputParser()


def format_docs(docs):
    """Convert retrieved LangChain documents into plain text for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def ask(question, test_number):
    """Retrieve relevant documents, show them, then generate an answer."""
    print("\n" + "=" * 60)
    print(f"\nTest question {test_number}: {question}")

    # Retrieve: turn the question into a vector and search FAISS for similar text.
    retrieved_docs = retriever.invoke(question)

    print("\nRetrieved documents from FAISS:")
    for doc in retrieved_docs:
        print(f"- {doc.page_content}")

    # Generate: pass the retrieved documents and question into the prompt/model.
    answer = answer_chain.invoke(
        {
            "context": format_docs(retrieved_docs),
            "question": question,
        }
    )

    print("\nAnswer:", answer)


# 8. Numbered test questions
test_questions = [
    (1, "What is the Roman name of Zurich?"),
    (2, "Is Zurich the capital of Switzerland?"),
    (3, "Who is the prime minister of Switzerland?"),  # This answer is not in the documents.
]


# 9. Run the tests
for test_number, question in test_questions:
    ask(question, test_number)
