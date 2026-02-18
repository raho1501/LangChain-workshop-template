import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from RAG.V1.RagV1 import RAGV1

model_name = "microsoft/Phi-3.5-mini-instruct"

system_prompt =(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " Use the tool to help answer user queries and cite the sources that are used."
    )

ragchain = RAGV1(model_name=model_name, system_prompt=system_prompt, document_base_dir="info-documents")

if __name__ == '__main__':
    ragchain.run_console()