# ChattyPDF

ChattyPDF is an application powered by LangChain and OpenAI, built using Streamlit, enabling users to interact with an LLM-powered chatbot. This bot is designed to handle various tasks related to text processing, PDF handling, image uploading, and question-answering.

## Prerequisites

- Python 3.7 or later
- Streamlit
- PyPDF2
- LangChain
- OpenAI's GPT-3.5
- MongoDB
- Pinecone
- Additional libraries specified in the `requirements.txt` file

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/ChattyPDF.git
    cd ChattyPDF
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Setup

1. Ensure you have the necessary API keys and configurations for the following services:
   - MongoDB: Replace `st.secrets["mongo"]` and configure MongoDB accordingly.
   - Pinecone: Replace `st.secrets["pinecone"]` and set up your Pinecone account.

2. Run the application:

    ```bash
    streamlit run chatty_pdf.py
    ```

## Usage

- Upon running the Streamlit app, the sidebar offers different options:
    - Upload PDFs: Upload PDF files to process and store their content for querying.
    - Partially delete PDF: Remove specific content from a previously uploaded PDF.
    - Ask Question: Interact with the chatbot to get answers related to uploaded PDFs.
    - Upload Image: Add images along with titles and descriptions for retrieval.
    - Delete File: Remove stored data associated with uploaded PDFs or images.

## Functionality Overview

- **Upload PDFs**:
    - Upload one or multiple PDF files.
    - Extract text content from uploaded PDFs and store it in Pinecone index.
- **Partially delete PDF**:
    - Delete specific content from a PDF stored in the index.
- **Ask Question**:
    - Interact with the chatbot to ask questions about the uploaded PDFs.
- **Upload Image**:
    - Upload images with titles and descriptions for retrieval.
- **Delete File**:
    - Delete stored data associated with uploaded PDFs or images.

## Contributors

- [Shreya Shrivastava](https://www.linkedin.com/in/shreya-shrivastava-b39911244/)
