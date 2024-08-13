
import json
import torch
from transformers import pipeline
from ragatouille import RAGPretrainedModel

if torch.cuda.is_available():
    torch.set_default_device("cpu")
    print("CUDA is available!!")
else:
    raise RuntimeError("CUDA is not available!! LLM cannot run, rerun with GPU")

class DemoInterface:
    def __init__(self,
                 json_path,
                 model_name='deepset/roberta-base-squad2-distilled',
                 retriever_model_name='colbert-ir/colbertv2.0'):
        self.raw_text_list = self.load_knowledge(json_path)
        self.model_name = model_name
        self.retriever_model_name = retriever_model_name
        self.retriever = None
        self.generator = None

        self.load_models()

    def load_knowledge(self, json_path):
        with open(json_path, 'r') as f:
            raw_text_json = json.load(f)
        raw_text_list = [text for _, text in raw_text_json.items()]
        if not isinstance(raw_text_list, list):
            raise TypeError("Expected raw_text_list to be a list.")
        return raw_text_list

    def load_models(self):
        self.retriever = RAGPretrainedModel.from_pretrained(self.retriever_model_name)
        print(type(self.raw_text_list))
        self.retriever = self.retriever.from_index(index_path='.ragatouille/colbert/indexes/knowledgestore_index/')
        #self.retriever.index(index_name="n_knowledgestore_index", collection=self.raw_text_list, use_faiss=True)
        self.generator = pipeline("question-answering", model=self.model_name)

    def ask(self, query):
        retrieved_docs = self.retrieve(query)
        if not retrieved_docs:
            return "No relevant information found."
        
        context = retrieved_docs[0]['content']
        return self.generate_response(context, query), context

    def retrieve(self, query, k=1):
        return self.retriever.search(query, k=k)

    def generate_response(self, context, query):
        response = self.generator(question=query, context=context)
        return response.get('answer', 'I don\'t know')

    @staticmethod
    def generate_prompt_fewshot(context, question):
        prompt_template = """
        You are an expert in understanding and interpreting provided text contexts. Given a context and a question, your task is to generate an accurate and informative answer based on the provided context. Here is the structure:

        1. **Context:** The detailed text or passage that contains the information needed to answer the question.
        2. **Question:** A specific question that needs to be answered based on the context.

        Please make sure your response is clear, concise, and directly addresses the question. If the context does not contain sufficient information to answer the question, say I don't know.

        **Example:**

        **Context:**
        "The rainforests of the Amazon are home to a vast diversity of species, including numerous plants, animals, and insects. These forests play a crucial role in regulating the Earth's climate by absorbing carbon dioxide and releasing oxygen. However, deforestation poses a significant threat to these ecosystems, leading to loss of habitat and biodiversity."

        **Question:**
        "Why are the rainforests of the Amazon important for the Earth's climate?"

        **Answer:**
        "The rainforests of the Amazon are important for the Earth's climate because they absorb carbon dioxide and release oxygen, helping to regulate the climate."

        Please follow this format for each question:

        **Context:**
        {context}

        **Question:**
        {question}

        **Final Answer:**
        """
        return prompt_template.format(context=context, question=question)

    @staticmethod
    def generate_prompt(context, query):
        prompt = f"""Give the answer to the user query delimited by triple backticks ```{query}```
                    using the information given in context delimited by triple backticks ```{context}```.
                    If there is no relevant information in the provided context, tell user that you did not have any relevant context to base your answer on. Be concise and output the answer.
                    """
        return prompt
