# Legal-Link-AI

Legal-Link-AI is a FastAPI-based application designed to manage and interact with legal documents. It offers functionalities such as document embedding, querying, highlighting, and text comparison. The project leverages Pinecone for vector storage and OpenAI for language processing, allowing users to query or chat with documents across multiple databases.

## Technology Stack

- **Python**
- **FastAPI**
- **OpenAI**
- **Pinecone**
- **MySQL**
- **Langchain**

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- OpenAI API key
- Pinecone API key
- MySQL database
- Langchain

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/moogleLabsDev/Legal_Link_AI.git
   cd Legal-Link-AI
   ```

2. **Create a virtual environment and activate it:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory with the following variables:
   ```sh
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   DATABASE_URL=mysql+pymysql://user:password@localhost/dbname
   ```

5. **Run the application:**
   ```sh
   uvicorn app.main:app --reload
   ```
