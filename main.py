import os.path, os, pinecone, fitz, shutil
from pathlib import Path
from pinecone import Pinecone, PodSpec
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
PINECONE = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
# PINECONE_INDEX = pinecone.Index(os.environ.get('PINECONE_API_KEY'), os.environ.get('PINECONE_HOST'))
PINECONE_INDEX = PINECONE.Index('RAG', host=os.environ.get('PINECONE_HOST'))
PINE_VECTOR_STORE = PineconeVectorStore(pinecone_index=PINECONE_INDEX)
BUSY = False

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


##======================================================== Basic Functions ============================================================================
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
        nodeList = gen_embedding(doc, nodeList)
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


def gen_embedding(doc, nodes):
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
    while True:
        query = input("Question: ")
        if query == 'exit':
            running = False
            break
        llmquery = query_engine.query(query)
        print(llmquery.response)


def reset():
    global PINECONE_INDEX
    try:
        # PINECONE_INDEX.delete(delete_all=True,namespace='Default')
        PINECONE.delete_index(PINECONE_INDEX_NAME)
        print("Database has been reset.")
    except:
        print("Error! Database already empty!")
    PINECONE.create_index(PINECONE_INDEX_NAME, dimension=1536, metric="cosine", spec=PodSpec(environment="gcp-starter"))
    PINECONE_INDEX = PINECONE.Index(name=PINECONE_INDEX_NAME)


# ==========================================   Gradio Interfaces and Functions =================================================
# Gradio querying function
def gradioQuery(message, history):
    index = VectorStoreIndex.from_vector_store(PINE_VECTOR_STORE)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    query_engine = RetrieverQueryEngine(retriever=retriever)
    if BUSY:
        return "Database is being updated. Please wait"
    if message == 'exit':
        return ' '
    llmquery = query_engine.query(message)
    return str(llmquery)


# Adding file into the vector database
def addFile(file):
    global PINE_VECTOR_STORE
    nodeList = []
    passednodes = []
    for x in range(len(file)):
        filePath = file[x].name
        fileName = os.path.basename(filePath)
        copyPath = DOCUMENT_PATH.name + '/' + fileName
        shutil.copyfile(filePath, copyPath)
        doc = fitz.open(filePath)
        nodeList = gen_embedding(doc, nodeList)
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
    global BUSY
    if file:
        BUSY = True
        filePath = DOCUMENT_PATH.name + '/' + file
        os.remove(filePath)
        reset()
        ingestion()
        BUSY = False
        return "File has been deleted!"
    else:
        return "File not found"


def deleteAll():
    global BUSY
    BUSY = True
    files = DOCUMENT_PATH.rglob('*.pdf')
    for file in files:
        filePath = DOCUMENT_PATH.name + '/' + file.name
        os.remove(filePath)
    reset()
    ingestion()
    BUSY = False
    return "Database has been emptied!"


def interfaces():
    with gr.Blocks() as demo:
        with gr.Tab("Chat Interface"):
            with gr.Column():
                gr.Markdown("### Chat")
                chat_history = gr.TextArea(label="Chat History", interactive=False, value="", lines=10)
                user_input = gr.Textbox(label="Your Message")
                send_button = gr.Button("Send")

                def update_chat_history(message, chat_history):
                    new_chat_history = chat_history + "\nUser: " + message + "\nBot: " + gradioQuery(message, chat_history)
                    return new_chat_history, ""

                send_button.click(
                    fn=update_chat_history,
                    inputs=[user_input, chat_history],
                    outputs=[chat_history, user_input]
                )
        with gr.Tab("File Management"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Upload Files")
                    file_input = gr.File(label="Upload file", file_count='multiple')
                    save_button = gr.Button("Save File")
                with gr.Column():
                    gr.Markdown("### Files")
                    files_output = gr.TextArea()
                    list_button = gr.Button("List Files")

            file_list = gr.Dropdown(label="Select a file to delete", choices=listFile())
            delete_button = gr.Button("Delete Selected File")
            delete_all_button = gr.Button("Destroy Database")
            save_button.click(addFile, inputs=file_input, outputs=files_output)
            list_button.click(listFile, inputs=[], outputs=files_output)
            delete_button.click(deleteFile, inputs=file_list, outputs=files_output)
            delete_all_button.click(deleteAll, inputs=[], outputs=files_output)
    return demo


# ========================================================== Main ==========================================================
def main():
    # init_pine()
    # reset()
    # ingestion()
    # retrieval()
    demo = interfaces()
    demo.launch()


if __name__ == '__main__':
    main()
