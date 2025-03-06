import logging
from langchain_core.runnables import RunnableLambda,RunnableBranch
from src.vector_store import VectorStore
from src.template import Template

logging.basicConfig(level=logging.DEBUG)

class Chaining:

    def __init__(self, model, memory, max_messages=5):
        self.model = model
        self.memory = memory
        self.max_messages = max_messages
        self.template = Template()
        self.vector = VectorStore()

    def get_recent_messages(self):
        """Return only the last `max_messages` from memory."""
        if hasattr(self.memory, "messages"):
            return self.memory.messages[-self.max_messages:]
        return []  # If no messages exist
    

    def not_related(self, query_data):
        prompt = self.template.not_related_template()
        query = query_data["query"]

        chain = prompt | self.model
        result = chain.invoke({"query": query, "chat_history": self.get_recent_messages()})  

        input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
        output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0

        logging.debug(f"Input Token: {input_tokens}\nOutput Token: {output_tokens}")
        logging.debug(f"Raw Model Response (Not Related): '{result}'")

        self.vector.save_token_usage_to_yaml(input_tokens,output_tokens)
        #Fallback response if the model fails to generate output
        response_text = result.content if result.content else "I'm not sure how to respond to that. Can you clarify?"

        return {
            "response": response_text,
        }


    def book_task_classifier(self, query_data):
        query = query_data["query"].strip().lower()  # Normalize input
        task_keywords = self.vector.load_yaml_as_dict()
        if not query:
            logging.error("Error: Received an empty query!")
            return {"error": "Query cannot be empty.", "book_task": "other", "chat_history": self.get_recent_messages()}

        #CLassification based on the keywords
        if any(word in query for word in task_keywords['tasks']['recommendation']):
            book_task = "recommendation"
        elif any(word in query for word in task_keywords['tasks']['available']):
            book_task = "available"
        elif any(word in query for word in task_keywords['tasks']['borrow']):
            book_task = "borrow"
        elif any(word in query for word in task_keywords['tasks']['return']):
            book_task = "return"
        elif any(word in query for word in task_keywords['tasks']['talk']):
            book_task = "talk"
        elif any(word in query for word in task_keywords['tasks']['history']):
            book_task = "history"
        elif any(word in query for word in task_keywords['tasks']['token']):
            book_task = "token"
        else:
            # Fallback to LLM-based classification if no rule matches
            prompt = self.template.book_query_classifier_template()

            chain = prompt | self.model
            result = chain.invoke({"query": query, "chat_history": self.get_recent_messages()})
            logging.debug(f"Raw Classifier Output: {result}")

            input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
            output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0
            
            self.vector.save_token_usage_to_yaml(input_tokens,output_tokens)

            book_task = result.content.strip().lower() if isinstance(result, str) else "other"
            if book_task not in ["recommendation", "talk", "other", "available", "borrow"]:
                book_task = "other"



        return {
            "query": query,
            "book_task": book_task,
        }


    def book_recommend(self,query_data):
        prompt = self.template.book_recommend_template()
        query = query_data["query"]
        
        chain = prompt | self.model
        context = self.vector.mmr_search(query)
        result = chain.invoke({"query": query, "chat_history": self.get_recent_messages() , "context": context})

        input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
        output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0

        self.vector.save_token_usage_to_yaml(input_tokens,output_tokens)


        logging.debug(f"Book Recommendation Result: {result}")

        return {
            "response": result.content,
        }

    def book_talk(self,query_data):
        query = query_data["query"] 
        prompt = self.template.book_talk_template()

        chain = prompt | self.model
        result = chain.invoke({"query": query, "chat_history":self.get_recent_messages()})

        input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
        output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0

        self.vector.save_token_usage_to_yaml(input_tokens,output_tokens)

        logging.debug(f"Book Talk: {query_data["book_task"]}+{result}")

        return {
            "response": result.content,
        }

    def book_availability(self,query_data):
        db = VectorStore().create_vector_store()  # Load the vector store
        prompt = self.template.book_availability_template()
        query = query_data["query"]

        context_docs = db.similarity_search(query, k=5)  # Use similarity search for better results

        # Debug retrieved documents
        logging.debug(f"Retrieved Book Docs: {[doc.page_content for doc in context_docs]}")

        # Prepare context for the model
        context = "\n".join(doc.page_content for doc in context_docs) if context_docs else "No matching books found."

        # Invoke the model with context
        chain = prompt | self.model
        result = chain.invoke({"query": query, "chat_history": self.get_recent_messages(), "context": context})

        input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
        output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0

        self.vector.save_token_usage_to_yaml(input_tokens,output_tokens)

        # Debugging: Log the raw response from the model
        logging.debug(f"Raw Model Response (Availability Check): {result}")

        return {
            "response": result.content,
        }
    
    def borrow_book(self,query_data):
        query = query_data["query"] 
        prompt = self.template.borrow_book_template()

        chain = prompt | self.model
        result = chain.invoke({"query": query, "chat_history":self.get_recent_messages()})

        input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
        output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0

        self.vector.save_token_usage_to_yaml(input_tokens,output_tokens)

        logging.debug(f"Book Borrow: {result}")

        return {
            "response": result.content,
        }
    
    def return_book(self,query_data):
        query = query_data["query"] 
        prompt = self.template.return_book_template()

        chain = prompt | self.model
        result = chain.invoke({"query": query, "chat_history":self.get_recent_messages()})

        input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
        output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0

        self.vector.save_token_usage_to_yaml(input_tokens,output_tokens)

        logging.debug(f"Book Return: {result}")

        return {
            "response": result.content,
        }
    
    def check_hisotry(self,query_data):
        query = query_data["query"]
        prompt = self.template.history_template()

        chain = prompt | self.model
        result = chain.invoke({"query": query, "chat_history":self.get_recent_messages()})

        input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
        output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0

        self.vector.save_token_usage_to_yaml(input_tokens,output_tokens)

        logging.debug(f"History: {result}")  

        return {
            "response": result.content
        } 
    
    def get_total_cost(self):
        token_consumption = self.vector.load_token_consumption()
        input_tokens = token_consumption['tokens']['input_tokens']
        output_tokens = token_consumption['tokens']['output_tokens']
        prompt =  self.template.token_cost_template()
        chain = prompt | self.model
        result = chain.invoke({"input_tokens": input_tokens, "output_tokens": output_tokens, "chat_history": self.get_recent_messages()})

        logging.debug(f"Result: {result}")
        return {
            "response": result.content
        }
    
    def combine_branch(self,query,chat_history):

        not_related_chain = RunnableLambda(lambda x: self.not_related(x) or {"query": x["query"], "response": "Sorry, I couldn't process that request."})
        book_recommend_chain = RunnableLambda(lambda x: self.book_recommend(x))

        #branch for the book task
        book_talk_chain = RunnableLambda(lambda x: self.book_talk(x))
        book_availability_chain = RunnableLambda(lambda x: self.book_availability(x))

        #Chain for borrow and return
        book_borrow_chain = RunnableLambda(lambda x: self.borrow_book(x))
        book_return_chain = RunnableLambda(lambda x: self.return_book(x))
        check_history_chain = RunnableLambda(lambda x: self.check_hisotry(x))
        token_cost_chain = RunnableLambda(lambda x: self.get_total_cost())
        book_task_branch = RunnableBranch(
            (lambda x: "recommendation" in x["book_task"], book_recommend_chain),
            (lambda x: "talk" in x["book_task"], book_talk_chain),
            (lambda x: "return" in x["book_task"],book_return_chain),
            (lambda x: "borrow" in x["book_task"],book_borrow_chain),
            (lambda x: "available" in x["book_task"], book_availability_chain),
            (lambda x: "history" in x["book_task"], check_history_chain),
            (lambda x: "token" in x["book_task"],token_cost_chain),
            (lambda x: "other" in x["book_task"], not_related_chain),  # Ensures fallback chain
            RunnableLambda(lambda x: {"query": x["query"], "response": "Sorry, I can only help with book-related topics. Let me know if you have any book-related questions!"})
        )

        
        book_related_chain = RunnableLambda(lambda x: self.book_task_classifier(x))
        chain =  book_related_chain | book_task_branch
        result = chain.invoke({"query": query, "chat_history": chat_history})

        return result