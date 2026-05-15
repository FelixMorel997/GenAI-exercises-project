import sys
import pysqlite3  # noqa: F401
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import os

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import Dict, List, Optional
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    # Criteria: must be a directory AND (name contains "chroma" OR contains typical Chroma persistence artifacts)
    
    root = Path(".").resolve()

    # 1) Look for ChromaDB directories by locating the persistence SQLite file.
    #    This avoids false positives like generic folders named "index".
    ignore_dir_names = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "data_text",  # raw documents folder (not a Chroma persistence folder)
    }

    chroma_dirs: List[Path] = []
    for sqlite_path in root.rglob("chroma.sqlite3"):
        # Skip anything inside ignored directories
        if any(part in ignore_dir_names for part in sqlite_path.parts):
            continue
        chroma_dirs.append(sqlite_path.parent)

    # De-duplicate (rglob can include nested dirs that match multiple criteria)
    seen = set()
    unique_dirs: List[Path] = []
    for d in chroma_dirs:
        d_resolved = str(d.resolve())
        if d_resolved not in seen:
            seen.add(d_resolved)
            unique_dirs.append(d)


    # Loop through each discovered directory
    for dir_path in unique_dirs:
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(dir_path),
                settings=Settings(anonymized_telemetry=False),
            )

            # Retrieve list of available collections from the database
            collections = client.list_collections()

            # If no collections, still register the backend directory (optional but helpful)
            if not collections:
                continue

            # Loop through each collection found
            for c in collections:
                collection_name = getattr(c, "name", str(c))

                # Create unique identifier key combining directory and collection names
                key = f"{dir_path.name}:{collection_name}"

                # Open the collection to query metadata like count
                try:
                    collection = client.get_collection(collection_name)
                    try:
                        doc_count = str(collection.count())
                    except Exception:
                        doc_count = "?"
                except Exception:
                    doc_count = "?"

                # Build information dictionary
                backends[key] = {
                    # Store directory path as string
                    "directory": str(dir_path),
                    # Store collection name
                    "collection_name": collection_name,
                    # Create user-friendly display name
                    "display_name": f"{dir_path.name} / {collection_name} ({doc_count} docs)",
                    # Get document count with fallback for unsupported operations
                    "doc_count": doc_count,
                }

        # Handle connection or access errors gracefully
        except Exception as e:
            # Create fallback entry for inaccessible directories
            err = str(e).replace("\n", " ").strip()
            if len(err) > 80:
                err = err[:77] + "..."

            key = f"{dir_path.name}:__error__"
            backends[key] = {
                "directory": str(dir_path),
                "collection_name": "",
                "display_name": f"{dir_path.name} (unavailable: {err})",
                "doc_count": "?",
            }

    # Return complete backends dictionary with all discovered collections
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    try:
        # Create a chomadb persistentclient
        # client = chromadb.PersistentClient(
        #     path=chroma_dir,
        #    settings=Settings(anonymized_telemetry=False),
        #)
        # Return the collection with the collection_name
        # collection=client.get_collection(name=collection_name)
        api_key = os.getenv("OPENAI_API_KEY", "")
        is_voc = api_key.startswith("voc")

        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Attach the SAME embedding model you used when building the DB
        # (text-embedding-3-small -> 1536 dims)
        kwargs = {"api_key": api_key, "model_name": "text-embedding-3-small"}
        
        if is_voc:
            # chromadb versions differ: some accept api_base, not base_url
            kwargs["api_base"] = "https://openai.vocareum.com/v1"

        embedding_function = OpenAIEmbeddingFunction(**kwargs)
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

        return collection, True, None
    except Exception as e:
        return None, False, str(e)

    

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable to None (represents no filtering)
    where_filter = None

    # Check if filter parameter exists and is not set to "all" or equivalent
    # If filter conditions are met, create filter dictionary with appropriate field-value pairs
    if mission_filter and str(mission_filter).strip().lower() not in {"all", "any", "*"}:
        where_filter = {"mission": mission_filter}

    # Execute database query with the following parameters:
    results = collection.query(
        query_texts=[query],  # Pass search query in the required format
        n_results=n_results,  # Set maximum number of results to return
        where=where_filter,  # Apply conditional filter (None for no filtering, dictionary for specific filtering)
    )

    # Return query results to caller   
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # Initialize list with header text for context section
    context_parts: List[str] = ["RETRIEVED CONTEXT:"]

    # Loop through paired documents and their metadata using enumeration
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        meta = meta or {}
        # Extract mission information from metadata with fallback value
        mission = meta.get("mission", "unknown")
        # Clean up mission name formatting (replace underscores, capitalize)
        mission_clean = str(mission).replace("_", " ").strip().title()

        # Extract source information from metadata with fallback value
        source = meta.get("source", "unknown_source")

        # Extract category information from metadata with fallback value
        category = meta.get("document_category", "general_document")
        # Clean up category name formatting (replace underscores, capitalize)
        category_clean = str(category).replace("_", " ").strip().title()

        # Create formatted source header with index number and extracted information
        header = f"[Source {i}] Mission: {mission_clean} | Category: {category_clean} | Source: {source}"
        # Add source header to context parts list
        context_parts.append(header)

        # Check document length and truncate if necessary
        doc_str = (doc or "").strip()
        max_chars = 1200
        if len(doc_str) > max_chars:
            doc_str = doc_str[:max_chars].rstrip() + "..."

        # Add truncated or full document content to context parts list
        context_parts.append(doc_str)
        context_parts.append("")  # blank line between sources

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_parts).strip()

backends = discover_chroma_backends()
print(backends)
