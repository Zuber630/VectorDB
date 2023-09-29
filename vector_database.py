# Import necessary libraries and modules
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import pandas as pd

# Define a class for the Keyword Search Application
class KeywordSearchApp:
    def __init__(self, model_name, data_csv):
        # Initialize the FastAPI app
        self.app = FastAPI()
        self.model_name = model_name
        self.data_csv = data_csv
        self.db = None  # Initialize the FAISS index as None
        
        # Initialize the Sentence Transformer Embeddings model
        self.embedding_function = SentenceTransformerEmbeddings(model_name=self.model_name)

        # Load data from a CSV file into a Pandas DataFrame
        self.load_data()

    def load_data(self):
        # Read data from a CSV file into a Pandas DataFrame
        self.df = pd.read_csv(self.data_csv)
        self.list_of_documents = []
        
        # Iterate through each row in the DataFrame
        for index, row in self.df.iterrows():
            text_value = row['reserved_keywords']
            
            # Create a Document object for each text value
            self.list_of_documents.append(Document(page_content=text_value))

    def create_faiss_index(self):
        # Create a FAISS index using the loaded data and the Sentence Transformer model
        self.db = FAISS.from_documents(self.list_of_documents, self.embedding_function)
        
        # Save the FAISS index locally
        self.db.save_local("faiss_index")

    
    def load_index(self, index_name = "faiss_index"):
        embedding_function = SentenceTransformerEmbeddings(model_name=self.model_name)
        self.db = FAISS.load_local(index_name, embedding_function)
        return self.db 

    def query_database(self, request_data):
        if self.db is None:
            raise ValueError("FAISS index is not loaded. Call load_faiss_index() to load the index.")
        
        query_text = request_data['query_text']
        results = []

        # Perform similarity search in the FAISS index
        results_with_scores = self.db.similarity_search_with_score(query_text)

        # Extract the results with their scores
        for doc, score in results_with_scores:
            results.append(doc.page_content)

        return results

    def process_csv(self, input_csv_filename = "/home/em-gpu-01/llmproject/data/data.csv", output_csv_filename="output_data.csv"):
        def create_and_query_database(input_string):
            # Modify this function to perform your specific processing
            # In this example, we'll use the query_database method
            request_data = {'query_text': input_string}
            results = self.query_database(request_data)
            return str(results)

        # Read the input CSV file into a Pandas DataFrame
        input_df = pd.read_csv(input_csv_filename)

        # Iterate through each row in the input DataFrame
        for index, row in input_df.iterrows():
            input_string = row["input"]

            # Apply the provided process_function to the input string
            result = create_and_query_database(input_string)

            # Append the result at the end of the input string
            row["input"] = "Description: /n" + input_string + "Banned Words: /n" + result

        # Write the modified input DataFrame to the output CSV file
        input_df.to_csv(output_csv_filename, index=False)

    def start(self):
        # Define a FastAPI POST endpoint for loading data, creating the FAISS index, and saving the index.
        @self.app.post("/load_create_index/")
        async def load_create_index():
            try:
                # Load data from the CSV file
                self.load_data()

                # Create and save the FAISS index
                self.create_faiss_index()

                return {"message": "Data loaded, FAISS index created, and index saved successfully."}
            except Exception as e:
                return HTTPException(status_code=400, detail="Error loading data and creating index.")

        # Define a FastAPI POST endpoint for loading an existing FAISS index.
        @self.app.post("/load_existing_index/")
        async def load_existing_index():
            try:
                # Load the existing FAISS index
                self.load_index()
                return {"message": "Existing FAISS index loaded successfully."}
            except Exception as e:
                return HTTPException(status_code=400, detail="Error loading existing index.")

        # Define a FastAPI POST endpoint for querying the database.
        @self.app.post("/query_database/", response_model=list[str])
        async def query_endpoint(request: Request):
            try:
                request_data = await request.json()
                results = self.query_database(request_data)
                return results
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid JSON Type")

        # Define a FastAPI POST endpoint for process_csv.
        @self.app.post("/process_csv/")
        async def process_csv_endpoint():
            try:
                # Process an input CSV file and write the results to an output CSV file
                self.process_csv()
                return {"message": "CSV processing completed successfully."}
            except Exception as e:
                return HTTPException(status_code=400, detail="Error processing CSV.")

        # Import uvicorn and run the FastAPI app
        import uvicorn
        uvicorn.run(self.app, host="127.0.0.1", port=8020)
if __name__ == "__main__":
    # Create an instance of KeywordSearchApp
    keyword_search_app = KeywordSearchApp(model_name='google/flan-t5-base', data_csv='/home/em-gpu-01/llmproject/data/banned_keywords.csv')

    # Start the FastAPI application
    keyword_search_app.start()