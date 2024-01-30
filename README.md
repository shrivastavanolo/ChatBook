# ChatBook

ChatBook is a chatbot application powered by LangChain and OpenAI, designed to help users quickly understand the content of PDF files without reading the entire document. The application is built using Streamlit and provides an interactive interface for querying PDF content.

## Features

- **PDF Upload:** Users can upload PDF files to the application.
- **Text Extraction:** The uploaded PDF is processed to extract text content.
- **Text Splitting:** The text is split into smaller chunks for efficient processing.
- **Embeddings and Vector Store:** OpenAI embeddings are generated, and a vector store is created for similarity search.
- **Question Answering:** Users can ask questions related to the PDF content, and the chatbot generates responses.
- **Persistence:** Vector stores are saved to disk for optimized performance on repeated use.

## Getting Started

### Prerequisites

Make sure you have the following dependencies installed:

- Streamlit
- PyPDF2
- LangChain
- FAISS
- Dotenv

### Installation

```bash
pip install -r requirements.txt
```

### Usage

1. Set up your environment variables using the `.env` file.

2. Run the Streamlit app:

```bash
streamlit run chattypdf.py
```

3. Upload a PDF file, ask questions, and interact with the chatbot.

## Authors

- [Shreya Shrivastava](https://www.linkedin.com/in/shreya-shrivastava-b39911244/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Streamlit
- LangChain
- OpenAI
- PyPDF2
