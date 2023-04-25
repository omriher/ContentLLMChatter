from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline # import Tokenizer
import torch
import argparse
import textwrap
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DirectoryLoader
from langchain.docstore.document import Document

from langchain.embeddings.base import Embeddings
class EmbeddingsCls(Embeddings):
	def __init__(self, model, tokenizer):
		super().__init__()
		self.tokenizer = tokenizer
		self.model = model

	def embed(self, texts):
		embeddings = []
		for text in texts:
			input_ids = self.tokenizer.encode(text, return_tensors='pt')
			with torch.no_grad():
				outputs = self.model(input_ids)
				embedding = outputs.last_hidden_state.mean(dim=1)
			embeddings.append(embedding)
		return embeddings

	def embed_documents(self, documents):
		embeddings = []
		for doc in documents:
			input_ids = self.tokenizer.encode(doc, return_tensors='pt')
			with torch.no_grad():
				outputs = self.model(input_ids)
				embedding = outputs.hidden_states[-1].mean(dim=1).numpy().flatten()
			embeddings.append(embedding)
		return embeddings

	def embed_query(self, query):
		input_ids = self.tokenizer.encode(query, return_tensors='pt')
		with torch.no_grad():
			outputs = self.model(input_ids)
			embedding = outputs.hidden_states[-1].mean(dim=1).numpy().flatten()
		return embedding

def load_model(name):
	model_name = name

	print("[...] Loading Tokenizer for: {}".format(model_name))
	tokenizer = LlamaTokenizer.from_pretrained(model_name)
	print("[...] Loading Model: {}".format(model_name))

	base_model = LlamaForCausalLM.from_pretrained(model_name,
												  load_in_8bit=True,  # Required 8 bit, and not full, on T4
												  device_map='auto',
												  output_hidden_states=True,)
	print("[+] Model {} Loaded".format(model_name))
	return base_model, tokenizer
	
def create_pipeline(model, tokenizer):
	print("[...] Generating Pipeline.")
	pipe = pipeline(
		"text-generation",
		model=base_model,
		tokenizer=tokenizer,
		max_length=1024, 
		temperature=0.7,
		top_p=0.95,
		repetition_penalty=1.15
	)

	local_llm = HuggingFacePipeline(pipeline=pipe)
	print("[+] Pipeline generated")
	return local_llm

def create_vector_store(folder, embeddings_class):
	import os
	from langchain.document_loaders import UnstructuredPDFLoader
	from langchain.indexes import VectorstoreIndexCreator
	text_folder = folder
	print("[...] Generating loaders for files in {}...".format(text_folder))
	loader = DirectoryLoader(text_folder)
	print("[...] Generating VectorStore...")
	index = VectorstoreIndexCreator(embedding=embeddings_class,
									# vectorstore_kwargs={"persist_directory": "db"}
									).from_loaders([loader])
	print("[+] VectorStore Created Successfully")
	return index

def query(prompt, index, llm):
	query = prompt
	print("[?] Asking: {}".format(query))
	resp = index.query(query, llm=llm)
	print("[!] Answer: {}".format(resp))

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='ContentLLMChatter')
	parser.add_argument('-m', '--model', help='Model name from HuggingFace', required=True)
	parser.add_argument('-f', '--folder', help='Folder containing PDF files to analyze', required=True)
	parser.add_argument('-q', '--question', help='What question would you like the LLM to answer regarding the PDFs', required=True)
	args = parser.parse_args()

	folder = args.folder
	q = args.question
	model_name = args.model
	
	print("------------------------")
	print("ContentLLMChatter v0.01")
	print("     by Omri Herscovici")
	print("------------------------")
	print("[M] Model: {}".format(model_name))
	print("[F] Folder: {}".format(folder))
	print("[Q] Question: {}\n".format(q))
	
	base_model, tokenizer = load_model(model_name)
	llm = create_pipeline(base_model, tokenizer)
	embeddings_class = EmbeddingsCls(base_model, tokenizer)
	idx = create_vector_store(folder, embeddings_class)
	query(q, idx, llm)
