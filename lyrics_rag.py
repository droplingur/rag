import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================================================================
# LYRICS RAG DEMO
# =============================================================================
# This version uses lyrics.csv as the document source.
#
# The RAG flow:
#   1. Load song lyrics from a CSV file.
#   2. Turn each song into a document.
#   3. Optionally create summaries for each song.
#   4. Choose whether FAISS should index lyrics, summaries, or both.
#   5. Retrieve relevant documents above a similarity threshold.
#   6. Present the top matching songs ranked by similarity score.


PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "lyrics.csv"
SUMMARY_CSV_PATH = PROJECT_DIR / "lyrics_with_summaries.csv"
SCRIPT_NAME = Path(__file__).name
EMBEDDINGS_DIR = PROJECT_DIR / "embeddings" / "lyrics_rag"

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHAT_MODEL_NAME = "gpt-4o-mini"
TOP_K = 4
TOP_SONGS = 3
MIN_CHUNK_COSINE_SIMILARITY = 0.25
CHUNK_SIZE = 700
CHUNK_OVERLAP = 0
# Choose what gets vectorized into FAISS:
#   "lyrics" = raw lyric chunks only
#   "summaries" = generated song summaries only
#   "both" = raw lyric chunks and generated summaries together
RETRIEVAL_SOURCE = "both"

SUMMARY_CACHE_DIR = PROJECT_DIR / "summaries"
SUMMARY_COLUMN = "song_summary"
SUMMARY_CSV_COLUMNS = ["song_title", "primary_artist", "plain_lyrics", SUMMARY_COLUMN]
SUMMARY_PREVIEW_CHARS = 500

SHOW_CHUNK_EXAMPLES = True
CHUNK_EXAMPLES_TO_PRINT = 2
CHUNK_PREVIEW_CHARS = 350


load_dotenv(PROJECT_DIR / ".env")


def load_lyrics_documents(csv_path):
    """Load lyrics.csv and convert each song into a LangChain Document."""
    df = pd.read_csv(csv_path)

    # Keep only rows where lyrics exist.
    df = df[df["lyrics_found"]]
    df = df[["song_title", "primary_artist", "plain_lyrics"]].dropna()
    df = df.drop_duplicates(subset=["song_title", "primary_artist"]).reset_index(drop=True)

    documents = []
    for _, row in df.iterrows():
        title = row["song_title"]
        artist = row["primary_artist"]
        lyrics = row["plain_lyrics"]

        # The page_content is the full song document before chunking.
        page_content = f"Song: {title}\nArtist: {artist}\nLyrics:\n{lyrics}"

        # Metadata lets us print helpful source information later.
        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "song_title": title,
                    "primary_artist": artist,
                    "source_type": "lyrics",
                },
            )
        )

    return documents


def split_documents(documents):
    """Split song documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def documents_cache_key(documents, extra_text=""):
    """Create a stable key from prepared document content."""
    parts = [extra_text]
    for doc in documents:
        parts.append(doc.metadata.get("source_type", ""))
        parts.append(doc.metadata["song_title"])
        parts.append(doc.metadata["primary_artist"])
        parts.append(doc.page_content)

    cache_text = "\n".join(parts)
    return hashlib.sha256(cache_text.encode("utf-8")).hexdigest()[:12]


def summary_cache_path(documents):
    """Use a local JSON file so song summaries are not regenerated every run."""
    cache_key = documents_cache_key(documents, extra_text=CHAT_MODEL_NAME)
    return SUMMARY_CACHE_DIR / f"song_summaries_{cache_key}.json"


def summarize_song(doc, summary_model):
    """Create one short summary for a song."""
    summary_prompt = (
        "Summarize this song's lyrics for a music search system. "
        "Focus on themes, emotions, situations, and imagery. "
        "Include enough detail to help retrieve the song for thematic questions."
        f"{doc.page_content[:4000]}"
    )

    response = summary_model.invoke(summary_prompt)
    return response.content.strip()


def save_summaries_csv(csv_path, output_csv_path, summaries):
    """Create lyrics_with_summaries.csv without changing the original lyrics.csv."""
    df = pd.read_csv(csv_path)

    def lookup_summary(row):
        key = f"{row['song_title']}|{row['primary_artist']}"
        return summaries.get(key, "")

    summary_df = df[["song_title", "primary_artist", "plain_lyrics"]].copy()
    summary_df[SUMMARY_COLUMN] = df.apply(lookup_summary, axis=1)
    summary_df = summary_df[SUMMARY_CSV_COLUMNS]

    if output_csv_path.exists():
        existing_df = pd.read_csv(output_csv_path)
        if existing_df.fillna("").equals(summary_df.fillna("")):
            print(f"{output_csv_path} already has up-to-date summaries")
            return

    summary_df.to_csv(output_csv_path, index=False)
    print(f"Saved summaries to {output_csv_path}")


def load_summary_documents(output_csv_path):
    """Load summary documents from lyrics_with_summaries.csv."""
    df = pd.read_csv(output_csv_path)
    df = df[["song_title", "primary_artist", SUMMARY_COLUMN]].dropna()

    summary_documents = []
    for _, row in df.iterrows():
        title = row["song_title"]
        artist = row["primary_artist"]
        summary = row[SUMMARY_COLUMN]

        summary_documents.append(
            Document(
                page_content=f"Song: {title}\nArtist: {artist}\nSummary:\n{summary}",
                metadata={
                    "song_title": title,
                    "primary_artist": artist,
                    "source_type": "summary",
                },
            )
        )

    return summary_documents


def ensure_summaries_csv(documents, csv_path, output_csv_path):
    """Create or refresh lyrics_with_summaries.csv when summary retrieval is needed."""
    SUMMARY_CACHE_DIR.mkdir(exist_ok=True)
    cache_path = summary_cache_path(documents)

    if cache_path.exists():
        print(f"Loading song summaries from {cache_path}")
        summaries = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        print(f"Creating summaries for {len(documents)} songs...")
        summaries = {}
        # Only instantiated on a cache miss to avoid an unnecessary object when summaries already exist.
        summary_model = ChatOpenAI(model=CHAT_MODEL_NAME, temperature=0)

        for number, doc in enumerate(documents, start=1):
            title = doc.metadata["song_title"]
            artist = doc.metadata["primary_artist"]
            key = f"{title}|{artist}"
            summaries[key] = summarize_song(doc, summary_model)
            print(f"  Summarized {number}/{len(documents)}: {title} by {artist}")

        cache_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"Saved song summaries to {cache_path}")

    save_summaries_csv(csv_path, output_csv_path, summaries)


def build_or_load_summary_documents(documents, csv_path, output_csv_path=SUMMARY_CSV_PATH):
    """Prepare summary CSV, then return summary documents for retrieval."""
    ensure_summaries_csv(documents, csv_path, output_csv_path)
    return load_summary_documents(output_csv_path)


def choose_random_song_key(documents):
    """Pick one random song from a list of documents."""
    song_keys = sorted(
        {
            (doc.metadata["song_title"], doc.metadata["primary_artist"])
            for doc in documents
        }
    )
    return random.choice(song_keys)


def docs_for_song(documents, song_key):
    """Return documents that belong to one song."""
    selected_title, selected_artist = song_key
    return [
        doc
        for doc in documents
        if doc.metadata["song_title"] == selected_title
        and doc.metadata["primary_artist"] == selected_artist
    ]


def print_summary_example(summary_documents, song_key):
    """Print the summary for the same song used in the chunk example."""
    selected_docs = docs_for_song(summary_documents, song_key)
    if not selected_docs:
        return

    doc = selected_docs[0]
    title = doc.metadata["song_title"]
    artist = doc.metadata["primary_artist"]
    preview = doc.page_content[:SUMMARY_PREVIEW_CHARS].replace("\n", " ")

    if len(doc.page_content) > SUMMARY_PREVIEW_CHARS:
        preview += "..."

    print("\nSummary example:")
    print(f"Song: {title} by {artist}")
    print(f"Preview: {preview}")


def print_chunk_examples(chunks, song_key, examples_to_print=CHUNK_EXAMPLES_TO_PRINT):
    """Print chunks from one random song so students can see chunking."""
    selected_chunks = docs_for_song(chunks, song_key)
    if not selected_chunks:
        return

    selected_title, selected_artist = song_key
    print("\nChunk examples:")
    print(f"Random song: {selected_title} by {selected_artist}")

    for number, chunk in enumerate(selected_chunks[:examples_to_print], start=1):
        title = chunk.metadata["song_title"]
        artist = chunk.metadata["primary_artist"]
        preview = chunk.page_content[:CHUNK_PREVIEW_CHARS].replace("\n", " ")

        if len(chunk.page_content) > CHUNK_PREVIEW_CHARS:
            preview += "..."

        print(f"\nChunk {number}")
        print(f"Song: {title} by {artist}")
        print(f"Characters: {len(chunk.page_content)}")
        print(f"Preview: {preview}")


def choose_retrieval_source(args):
    """Decide whether FAISS should contain lyrics, summaries, or both."""
    return args.retrieval_source or RETRIEVAL_SOURCE


def build_retrieval_documents(lyrics_documents, csv_path, retrieval_source):
    """Prepare the exact documents that will be embedded into FAISS."""
    if retrieval_source not in {"lyrics", "summaries", "both"}:
        raise ValueError("retrieval_source must be 'lyrics', 'summaries', or 'both'")

    lyric_chunks = []
    summary_documents = []
    example_song_key = choose_random_song_key(lyrics_documents)

    if retrieval_source in {"lyrics", "both"}:
        lyric_chunks = split_documents(lyrics_documents)
        print(f"Split lyrics into {len(lyric_chunks)} chunks")

        if SHOW_CHUNK_EXAMPLES:
            print_chunk_examples(lyric_chunks, example_song_key)

    if retrieval_source in {"summaries", "both"}:
        summary_documents = build_or_load_summary_documents(lyrics_documents, csv_path)
        print(f"Using {len(summary_documents)} song summary documents")

        if SHOW_CHUNK_EXAMPLES:
            print_summary_example(summary_documents, example_song_key)

    if retrieval_source == "lyrics":
        return lyric_chunks
    if retrieval_source == "summaries":
        return summary_documents
    return lyric_chunks + summary_documents


def save_cache_metadata(cache_path, documents, csv_path, retrieval_mode):
    """Write a small note so we know which script created this FAISS index."""
    metadata = (
        f"script={SCRIPT_NAME}\n"
        f"source_csv={csv_path.name}\n"
        f"retrieval_mode={retrieval_mode}\n"
        f"embedding_model={EMBEDDING_MODEL_NAME}\n"
        f"document_count={len(documents)}\n"
        f"chunk_size={CHUNK_SIZE}\n"
        f"chunk_overlap={CHUNK_OVERLAP}\n"
    )
    (cache_path / "metadata.txt").write_text(metadata, encoding="utf-8")


def build_or_load_vectorstore(documents, embedding_model, csv_path, retrieval_mode):
    """Create a FAISS index once, then load it from disk on future runs."""
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    cache_key = documents_cache_key(
        documents,
        extra_text=(
            f"{retrieval_mode}:{EMBEDDING_MODEL_NAME}:cosine_normalized_l2:"
            f"{CHUNK_SIZE}:{CHUNK_OVERLAP}"
        ),
    )
    vectorstore_path = EMBEDDINGS_DIR / f"{retrieval_mode}_faiss_{cache_key}"

    if vectorstore_path.exists():
        print(f"Loading FAISS index from {vectorstore_path}")
        return FAISS.load_local(
            folder_path=str(vectorstore_path),
            embeddings=embedding_model,
            # This is safe here because we only load a local file created by this script.
            allow_dangerous_deserialization=True,
        )

    print(f"Creating FAISS index for {len(documents)} {retrieval_mode} documents...")
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model,
        normalize_L2=True,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    vectorstore.save_local(str(vectorstore_path))
    save_cache_metadata(vectorstore_path, documents, csv_path, retrieval_mode)
    print(f"Saved FAISS index to {vectorstore_path}")
    return vectorstore


def retrieve_docs_with_cosine_similarity(topic, vectorstore, k=TOP_K):
    """Search FAISS and return cosine similarity scores.

    FAISS returns squared L2 distance here. Because embeddings are normalized
    when indexed and queried, cosine similarity is 1 - distance / 2.
    """
    return vectorstore.similarity_search_with_score(query=topic, k=k)


def distance_to_cosine_similarity(distance):
    """Convert squared L2 distance between normalized vectors to cosine similarity."""
    return 1 - (distance / 2)


def top_songs_for_topic(topic, vectorstore, top_n=TOP_SONGS):
    """Return the best matching unique songs for a topic."""
    docs_with_distances = retrieve_docs_with_cosine_similarity(topic, vectorstore, k=top_n * 8)
    songs = {}

    for doc, distance in docs_with_distances:
        title = doc.metadata["song_title"]
        artist = doc.metadata["primary_artist"]
        source_type = doc.metadata.get("source_type", "lyrics")
        similarity = distance_to_cosine_similarity(distance)

        if source_type == "lyrics" and similarity < MIN_CHUNK_COSINE_SIMILARITY:
            continue

        key = (title, artist)

        if key not in songs or similarity > songs[key]["similarity"]:
            songs[key] = {
                "title": title,
                "artist": artist,
                "similarity": similarity,
                "source_type": source_type,
            }

    return sorted(songs.values(), key=lambda song: song["similarity"], reverse=True)[:top_n]


def print_top_songs(topic, vectorstore):
    """Print the top matching songs for a topic."""
    print("\n" + "=" * 60)
    print(f"\nSongs about: {topic}")

    songs = top_songs_for_topic(topic, vectorstore)
    if not songs:
        print(
            "\nNo matching songs found. "
            f"Lyric chunks below cosine similarity {MIN_CHUNK_COSINE_SIMILARITY:.2f} are ignored."
        )
        return

    print(f"\nTop {len(songs)} songs:")
    for number, song in enumerate(songs, start=1):
        print(
            f"{number}. {song['title']} by {song['artist']} "
            f"(cosine similarity {song['similarity']:.2f}, source {song['source_type']})"
        )


def build_vectorstore_for_source(lyrics_documents, csv_path, retrieval_source, embedding_model):
    """Build or load a FAISS index for a single retrieval source."""
    docs = build_retrieval_documents(lyrics_documents, csv_path, retrieval_source)
    return build_or_load_vectorstore(docs, embedding_model, csv_path, retrieval_source)


def print_compare_results(topic, vectorstores):
    """Print top songs for a topic across all three retrieval sources side by side."""
    print("\n" + "=" * 60)
    print(f"Comparing retrieval sources for: {topic}")

    for source, vectorstore in vectorstores.items():
        print(f"\n-- Source: {source} --")
        songs = top_songs_for_topic(topic, vectorstore)
        if not songs:
            print(
                f"  No matches above cosine similarity {MIN_CHUNK_COSINE_SIMILARITY:.2f}."
            )
            continue
        for number, song in enumerate(songs, start=1):
            print(
                f"  {number}. {song['title']} by {song['artist']} "
                f"(cosine similarity {song['similarity']:.2f}, matched via {song['source_type']})"
            )


def ask_topics_until_exit(vectorstore):
    """Let the user ask for several song topics without rebuilding FAISS."""
    print("\nType 'exit' or 'quit' to stop.")

    while True:
        topic = input("\nWhat do you want songs to be about? ").strip()

        if topic.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        if not topic:
            continue

        print_top_songs(topic, vectorstore)


def ask_topics_compare_mode(lyrics_documents, csv_path, embedding_model):
    """Interactive loop that compares lyrics, summaries, and both for every query."""
    print("\nBuilding all three FAISS indexes for comparison...")
    vectorstores = {
        source: build_vectorstore_for_source(lyrics_documents, csv_path, source, embedding_model)
        for source in ("lyrics", "summaries", "both")
    }
    print("\nType 'exit' or 'quit' to stop.")

    while True:
        topic = input("\nWhat do you want songs to be about? ").strip()

        if topic.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        if not topic:
            continue

        print_compare_results(topic, vectorstores)


parser = argparse.ArgumentParser(description="Find songs in lyrics.csv with RAG.")
parser.add_argument("--topic", help="Find songs about this topic and exit")
parser.add_argument("--question", help="Deprecated alias for --topic")
parser.add_argument(
    "--retrieval-source",
    choices=["lyrics", "summaries", "both"],
    help="Choose what to vectorize into FAISS: lyrics, summaries, or both",
)
parser.add_argument(
    "--compare",
    action="store_true",
    help="Run the same query against lyrics, summaries, and both — shows results side by side",
)
args = parser.parse_args()


# 1. Load raw lyrics from CSV.
lyrics_documents = load_lyrics_documents(CSV_PATH)
print(f"Loaded {len(lyrics_documents)} lyric documents from {CSV_PATH}")

embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

# 2. Compare mode builds all three indexes and lets the user query all three at once.
if args.compare:
    topic = args.topic or args.question
    if topic:
        vectorstores = {
            source: build_vectorstore_for_source(lyrics_documents, CSV_PATH, source, embedding_model)
            for source in ("lyrics", "summaries", "both")
        }
        print_compare_results(topic, vectorstores)
    else:
        ask_topics_compare_mode(lyrics_documents, CSV_PATH, embedding_model)
else:
    # 2. Document preparation comes first.
    # Depending on RETRIEVAL_SOURCE, this prepares:
    #   - lyric chunks
    #   - summary documents
    #   - both lyric chunks and summary documents
    # If summaries are needed, they are created and saved to lyrics_with_summaries.csv here,
    # before any FAISS index is created.
    retrieval_mode = choose_retrieval_source(args)

    print(f"Retrieval source: {retrieval_mode}")
    retrieval_documents = build_retrieval_documents(
        lyrics_documents,
        CSV_PATH,
        retrieval_mode,
    )

    # 3. Create or load the FAISS vector database after preparation is finished.
    vectorstore = build_or_load_vectorstore(
        retrieval_documents,
        embedding_model,
        CSV_PATH,
        retrieval_mode,
    )

    # 4. Ask what kind of songs the user wants.
    topic = args.topic or args.question
    if topic:
        print_top_songs(topic, vectorstore)
    else:
        ask_topics_until_exit(vectorstore)
