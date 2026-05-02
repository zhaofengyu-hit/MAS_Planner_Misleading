from typing import Annotated
import os
from ragflow_sdk import RAGFlow
from ragflow_sdk import Document
from autogen_core.tools import StaticWorkbench, FunctionTool
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("RAGFLOW_API")
if not api_key:
    raise ValueError("RAGFLOW_API environment variable is not set in .env file")
base_url = "http://localhost:9380"
dataset_name = "Temp"

rag = RAGFlow(api_key=api_key, base_url=base_url)


def upload_and_parse_files(
    list_of_file_path: Annotated[
        list[str], "List of local file paths to be uploaded and parsed."
    ],
) -> Annotated[str, "Parsing results for all uploaded or existing documents."]:
    """
    Upload multiple files to the dataset if not already present, trigger parsing for all relevant documents,
    and return the parsing status summary.
    """
    datasets = rag.list_datasets(name=dataset_name)
    dataset = datasets[0] if datasets else rag.create_dataset(name=dataset_name)

    upload_batch = []
    uploaded = []
    existing_docs = []

    # Check each file
    for file_path in list_of_file_path:
        display_name = os.path.basename(file_path)

        # Check if it already exists
        docs = dataset.list_documents(keywords=display_name, page=1, page_size=1)
        if len(docs) != 0:
            if docs[0].run.lower() == "done":
                existing_docs.append(docs[0])
            else:
                uploaded.append(docs[0])
            continue

        # Prepare for upload
        with open(file_path, "rb") as f:
            blob = f.read()
        upload_batch.append({"display_name": display_name, "blob": blob})

    # Upload and parse new files
    if upload_batch:
        uploaded.extend(dataset.upload_documents(upload_batch))

        id2doc = {doc.id: doc for doc in uploaded}
        doc_ids = list(id2doc.keys())

        parse_results = dataset.parse_documents(doc_ids)
        failed_docs = []
        for result in parse_results:
            doc_id, status = result[0], result[1]
            if status.lower() != "done":
                failed_docs.append(f"{id2doc[doc_id].name} (ID: {doc_id})")
        if failed_docs:
            raise Exception(f"Document parsing failed for: {', '.join(failed_docs)}")

    return "Upload and parsing completed."


upload_and_parse_tool = FunctionTool(
    upload_and_parse_files,
    description=(
        "Upload one or more local files to the RAGFlow dataset, where each document is saved "
        "by its file name (not full file path). "
        "If a document with the same name already exists, it will be skipped. "
        "Then trigger document parsing for new files and return parsing results. "
        "When using this tool, remember that subsequent lookups or retrievals must reference "
        "the document's file name, not its local path. "
        "PDF files are recommended for upload and parsing because this tool processes them efficiently. "
        "The tool also supports text files and image files, as well as common office formats (CSV, XLSX, PPTX, DOCX, etc.). "
        "However, when parsing office documents, only textual content is extracted; non-text features "
        "(e.g., shapes or symbols represented via colors in an XLSX file) are ignored. "
        "For such cases, specialized tools should be used instead."
    ),
)


def list_document_chunks_by_page(
    file_name: Annotated[str, "File name to retrieve."],
    page: Annotated[int, "Which page of chunks to retrieve (1-based)."],
    page_size: Annotated[int, "Number of chunks per page."],
) -> Annotated[str, "Chunk content of the specified page with page info."]:
    """
    List parsed chunks of a document sequentially by page.
    This function is for ordered browsing of document content,
    not semantic retrieval or relevance search.
    Returns chunk text with pagination info.
    """
    datasets = rag.list_datasets(name=dataset_name)
    if not datasets:
        raise Exception(f"Dataset '{dataset_name}' not found.")
    dataset = datasets[0]

    docs = dataset.list_documents(keywords=file_name, page=1, page_size=1)
    if not docs:
        raise Exception(f"No document found with name '{file_name}'")
    doc = docs[0]

    chunks = doc.list_chunks(page=page, page_size=page_size)
    total_chunks = doc.chunk_count
    total_pages = (total_chunks + page_size - 1) // page_size

    text = (
        f"Document: {doc.name} | Page {page}/{total_pages} | {len(chunks)} chunks\n\n"
    )
    for i, c in enumerate(chunks, 1):
        text += f"[Chunk {i}] {getattr(c, 'content', '')}\n\n"
    return text


list_chunks_by_page_tool = FunctionTool(
    list_document_chunks_by_page,
    description=(
        "List parsed document chunks sequentially by page number and page size, "
        "using the document's file name (not its local path) for lookup. "
        "This is intended for ordered reading or browsing through a document's content, "
        "not for semantic retrieval. "
        "Its retrieval efficiency is lower, but it ensures that no potential information is missed."
    ),
)


def retrieve_relevant_chunks(
    input: Annotated[str, "The string used to retrieve relevant chunks."],
    file_names: Annotated[list[str], "List of file names to restrict retrieval scope."],
    page: Annotated[int, "Page number of retrieval results (1-based)."],
    page_size: Annotated[int, "Number of chunks per page."],
) -> Annotated[
    str, "Text chunks most relevant to the input from the specified files."
]:
    """
    Retrieve document chunks that are semantically relevant to the given input,
    restricted to the specified file names. Only the given page of results is returned.
    """
    # Locate dataset
    datasets = rag.list_datasets(name=dataset_name)
    if not datasets:
        raise Exception(f"Dataset '{dataset_name}' not found.")
    dataset = datasets[0]

    # Collect document IDs for specified file names
    doc_ids = []
    for name in file_names:
        docs = dataset.list_documents(keywords=name, page=1, page_size=1)
        if not docs:
            continue
        doc_ids.append(docs[0].id)

    if not doc_ids:
        raise Exception(f"No matching documents found for {file_names}")

    # Execute retrieval within limited scope
    all_chunks = rag.retrieve(
        question=input,
        dataset_ids=[dataset.id],
        document_ids=doc_ids,
        page=page,
        page_size=page_size,
    )

    if not all_chunks:
        return f"No relevant chunks found for input: {input}"

    # Format output
    text = f"Input: '{input}' | Page {page} | {len(all_chunks)} chunks\n\n"
    for i, c in enumerate(all_chunks, 1):
        text += f"[Chunk {i}] {getattr(c, 'content', '')}\n\n"

    return text


retrieve_relevant_tool = FunctionTool(
    retrieve_relevant_chunks,
    description=(
        "Retrieve text chunks that are semantically relevant to a given string, "
        "restricted to the specified document file names (not file paths). "
        "Searches only within the named documents in the dataset. "
        "Use this for semantic similarity retrieval with pagination support. "
        "Use this for high-efficiency semantic similarity retrieval with pagination support, "
        "noting that the semantic filter may miss some potential information."
    ),
)


def get_workbench():
    FileParserWorkbench = StaticWorkbench(
        [upload_and_parse_tool, list_chunks_by_page_tool, retrieve_relevant_tool]
    )
    return FileParserWorkbench


# if __name__ == "__main__":
#     question = "Why is Dejah Thoris weeping when John Carter returns?"
#     result = retrieve_relevant_chunks("Why is Dejah Thoris weeping when John Carter returns?", ["afd1efe6-03dd-478c-9eb1-e562355ee94e.txt"], 1, 1)
#     print(repr(result))
