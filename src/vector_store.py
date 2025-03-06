from langchain_chroma import Chroma
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import yaml
import os

#Initialize the environment variable
load_dotenv()

class VectorStore:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.current_dir, "datasets", "KB_march3.csv")
        self.db_dir = os.path.join(self.current_dir,"db")
        self.persistent_directory = os.path.join(self.db_dir, "gemini_chroma_kb")

    # Load the YAML file into a dictionary
    def load_yaml_as_dict(self):
        yaml_path = os.path.join(self.current_dir, "yaml" ,"classifier.yaml")
        with open(yaml_path, "r") as file:
            data_dict = yaml.safe_load(file)
        return data_dict
    
    def load_token_consumption(self):
        yaml_path = os.path.join(self.current_dir, "yaml", "token_usage.yaml")
        with open(yaml_path, "r") as file:
            data_dict = yaml.safe_load(file)
        return data_dict
    
    def save_token_usage_to_yaml(self, input_tokens, output_tokens):
        """Append token usage to a YAML file."""
        TOKEN_USAGE_FILE = os.path.join(self.current_dir, "yaml", "token_usage.yaml")
        data = {"tokens": {"input_tokens": input_tokens, "output_tokens": output_tokens}}

        if os.path.exists(TOKEN_USAGE_FILE):
            with open(TOKEN_USAGE_FILE, "r") as file:
                existing_data = yaml.safe_load(file) or {}
                existing_tokens = existing_data.get("tokens", {}) or {}

                # Ensure existing tokens are integers (not None)
                existing_input_tokens = existing_tokens.get("input_tokens") or 0
                existing_output_tokens = existing_tokens.get("output_tokens") or 0

                data["tokens"]["input_tokens"] += existing_input_tokens
                data["tokens"]["output_tokens"] += existing_output_tokens

        with open(TOKEN_USAGE_FILE, "w") as file:
            yaml.dump(data, file)

        print(f"Total Input Tokens Used: {data['tokens']['input_tokens']}")
        print(f"Total Output Tokens Used: {data['tokens']['output_tokens']}")


    def split_docs(self):
        if os.path.exists(self.file_path):
            print("Loading the CSV File")
            loader = CSVLoader(self.file_path)
            docs = loader.load()
            print("Split the text")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_overlap=200,
                chunk_size=1500
            )
            chunk_docs = text_splitter.split_documents(docs)
            print("Successfully create a chunks documents")
            return chunk_docs
        else:

            raise FileNotFoundError(f"The file does not exist {self.file_path}")
    
    def create_vector_store(self):
        if not os.path.exists(self.persistent_directory):
            print("---Persistent does not exist yet. Creating new vector store ---")
            docs = self.split_docs()
            db = Chroma.from_documents(
                docs,self.embeddings,persist_directory=self.persistent_directory
            )
            print("--- Finished Creating Vector Store ---")

            return db
        else:
            print("Vector store already exists. Loading ......")
            db = Chroma(
                persist_directory=self.persistent_directory,
                embedding_function=self.embeddings
            )
            return db

    def mmr_search(self, query):
        db = self.create_vector_store()

        context_docs = db.max_marginal_relevance_search(query, k=5, fetch_k=30, lambda_mult=0.5)

        # Generate final context
        context = "\n".join(doc.page_content for doc in context_docs)

        return context


