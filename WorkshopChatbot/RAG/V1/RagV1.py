import os
import torch
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from RAG.rag import RAG
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PDFMinerLoader, UnstructuredMarkdownLoader
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from .RetrieveDocumentsMiddleware import RetrieveDocumentsMiddleware
import traceback


class RAGV1(RAG):
    def __init__(self, model_name, system_prompt, document_base_dir, add_email_masking = False):

        self.pipeline = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            model_kwargs=dict(
                device_map="cuda" if torch.cuda.is_available() else "cpu", 
                dtype="auto", 
                trust_remote_code=False
            ),
            pipeline_kwargs=dict(
                max_new_tokens=1024,
                return_full_text=False,
                temperature=0.2,
            ),
        )
        
        self.llm = ChatHuggingFace(llm=self.pipeline)

        self.system_prompt = system_prompt
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        embedding_dim = len(embeddings.embed_query(system_prompt))
        index = faiss.IndexFlatL2(embedding_dim)

        self.vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            )

        chunks = []
        for file in os.listdir(document_base_dir):
            if file.endswith(".txt") or file.endswith(".md"):
                docs = TextLoader(os.path.join(document_base_dir, file)).load()
            elif file.endswith(".pdf"):
                docs = PDFMinerLoader(os.path.join(document_base_dir, file)).load()
            elif file.endswith(".md"):
                docs = UnstructuredMarkdownLoader(os.path.join(document_base_dir, file)).load()
            else:
                print(f"Unsupported file type: {file}, skipping...")
                continue
            file_chunks = splitter.split_documents(docs)
            chunks.extend(file_chunks)

        self.vector_store.add_documents(chunks)
        self.checkpointer = None
        self.middlewares = []

        self.middlewares.append(RetrieveDocumentsMiddleware(self.vector_store, 0.0))

        if add_email_masking:
            self.middlewares.append(PIIMiddleware("email", strategy="redact", apply_to_output=True))

        self.agent = create_agent(
            self.llm, 
            tools=[], 
            middleware= self.middlewares,
            checkpointer=self.checkpointer,
            system_prompt=system_prompt)

    def run_console(self):
        messages = []
        print("Please chat...")
        while True:
            try:
                question = input(">>> ")
                message_payload = {"messages": [{"role": "user", "content": question }]}
                messages.append({"role": "user", "input": question})
                reply = ''
                try:
                    response = self.agent.invoke(message_payload, stream_mode="values")
                    print("AI: >>> ", end="\t")
                    print(response["messages"][-1].content)
                    reply = response["messages"][-1].content
                except Exception:
                    reply = traceback.format_exc()
                    print(reply)
                    print('Sorry, i cant reply to that', end='', flush=True)
                print()
                messages.append({"role": "assistant", "content": reply})
            except KeyboardInterrupt:
                break
        
        print("Exiting chat...")