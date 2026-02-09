# LangChain-workshop-template
Starting template for the workshop scheduled for the 18th of February, 2026.
In this workshop we will learn how to create a simple chatbot which supports Retrieval Augmented Generation using the LangChain framework.

# Install
- Create a virtual environment for development
  - Use `requirements_gpu.txt` if you have cuda installed
  - Otherwise, use `requirements_cpu.txt` for your environment
- Download the pretrained models that we will be using during our workshop.
  - We will be using the model (Phi-3.5-mini-instruct)[https://huggingface.co/microsoft/Phi-3.5-mini-instruct] from microsoft which is under a MIT-license for our LLM
    - Use the huggingface cli to download it by running `hf download microsoft/Phi-3.5-mini-instruct` in the console
  - We will be using the embedding model (sentence-transformers/all-mpnet-base-v2)[https://huggingface.co/sentence-transformers/all-mpnet-base-v2] to create embeddings which we then store inside an in-memory vectorstore
    - Use the huggingface cli to download it by running `hf download sentence-transformers/all-mpnet-base-v2` in the console
- Now you are ready to join the workshop.