import os.path, os, pinecone, fitz,shutil
from pathlib import Path
from pinecone import Pinecone
from llama_index.core import (
    VectorStoreIndex,
    set_global_service_context,
    ServiceContext,
)
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import gradio as gr
PINECONE_INDEX_NAME = 'rag'
PINECONE_INDEX = pinecone.Index(os.environ.get('PINECONE_API_KEY'), os.environ.get('PINECONE_HOST'))
PINE_VECTOR_STORE = PineconeVectorStore(pinecone_index=PINECONE_INDEX)

DOCUMENT_PATH = Path('./documents')
embed_model = OpenAIEmbedding()
text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)
nodes = []
llm = OpenAI(model="gpt-4")

service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context)

# Initialises the database
def init_pine():
    # Pinecone
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        raise FileNotFoundError("Index not found, check your index name.")

# Initial add of the entire folder into vector database. DO ONCE ONLY
def ingestion():
    passednodes = []
    nodeList = []
    global PINE_VECTOR_STORE
    for file in DOCUMENT_PATH.rglob('*.pdf'):
        doc = fitz.open(file)
        nodeList = gen_embedding(doc,nodeList)
    for node in nodeList:
        print('=====================')
        print(node)
        try:
            node_embedding = embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding
            passednodes.append(node)
        except:
            pass
    PINE_VECTOR_STORE.add(passednodes)


def gen_embedding(doc,nodes):
    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata in (3)
    doc_idxs = []
    for doc_idx, page in enumerate(doc):
        page_text = page.get_text("text")
        cur_text_chunks = text_parser.split_text(page_text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc_idx = doc_idxs[idx]
        src_page = doc[src_doc_idx]
        nodes.append(node)
    return nodes

# CLI code for querying GPT
def retrieval():
    index = VectorStoreIndex.from_vector_store(PINE_VECTOR_STORE)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    query_engine = RetrieverQueryEngine(retriever=retriever)
    running = True
    while running:
        query = input("Question: ")
        if query == 'exit':
            running = False
            break
        llmquery = query_engine.query(query)
        print(llmquery.response)

def reset():
    try:
        PINECONE_INDEX.delete(delete_all=True)
        print("Database has been reset.")
    except:
        print("Error! Database already empty!")

# ==========================================   Gradio Interfaces and Functions =================================================
# Gradio querying function
def gradioQuery(message,history):
    index = VectorStoreIndex.from_vector_store(PINE_VECTOR_STORE)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    query_engine = RetrieverQueryEngine(retriever=retriever)
    if message == 'exit':
        return ' '
    llmquery = query_engine.query(message)
    return str(llmquery)

# Adding file into the vector database
def addFile(file):
    global PINE_VECTOR_STORE
    filePath = file.name
    fileName = os.path.basename(filePath)
    copyPath = DOCUMENT_PATH.name+'/'+fileName
    shutil.copyfile(filePath,copyPath)
    nodeList = []
    doc = fitz.open(filePath)
    nodeList = gen_embedding(doc,nodeList)
    passednodes = []

    for node in nodeList:
        print('=====================')
        print(node)
        try:
            node_embedding = embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding
            passednodes.append(node)
        except:
            pass
    print(PINE_VECTOR_STORE)
    PINE_VECTOR_STORE.add(passednodes)
    return "File Inserted. On to the next!", listFile()
# except:
#     return "Error during file handling :(", listFile()

def listFile():
    fileList = []
    for file in DOCUMENT_PATH.rglob('*.pdf'):
        fileName = os.path.basename(file)
        fileList.append(fileName)
    return fileList

def deleteFile(file):
    filePath = DOCUMENT_PATH.name+'/'+file
    os.remove(filePath)
    reset()
    ingestion()
    return "File has been deleted!"

def interfaces():
    # Create the file upload interface.
    uploadInterface = gr.Interface(
        fn=addFile,
        inputs=gr.File(label="Upload your file here"),
        outputs=["text","text"],
        title="File Upload"
    )
    chatInterface = gr.ChatInterface(gradioQuery)

    # Combine both interfaces into tabs.
    tabInterface = gr.TabbedInterface(
        interface_list=[chatInterface, uploadInterface],
        tab_names=["Chat", "File Upload"]
    )
    return tabInterface

# ========================================================== Main ==========================================================
def main():
    init_pine()
    # reset()
    # ingestion()
    # retrieval()
    demo = interfaces()
    demo.launch()

if __name__ == '__main__':
    main()