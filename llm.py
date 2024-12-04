import os
import requests
import json_repair
from dotenv import load_dotenv

load_dotenv(".env")

MISTRAL_API_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_EMBED_URL = "https://api.mistral.ai/v1/embeddings"


def mistral_request(messages, model, **kwargs):
	api_key = os.getenv("MISTRAL_API_KEY")
	headers = {
		"Accept": "application/json",
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json"
	}
	data = {
		"model": model,
		"messages": messages,
		**kwargs
	}
	
	for tries in range(5):
		response = requests.post(MISTRAL_API_CHAT_URL, json=data, headers=headers)
		if response.ok:
			break
		elif response.status_code == 429:
			wait_time = 2 ** tries
			#print(f"Waiting {wait_time} second(s)...")
			time.sleep(wait_time)
		else:
			print(response.text)
			response.raise_for_status()
	else:
		print(response.text)
		response.raise_for_status()
	
	
	return response.json()
	
def mistral_embed_texts(inputs):
	api_key = os.getenv("MISTRAL_API_KEY")
	headers = {
		"Accept": "application/json",
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json"
	}
	data = {
		"model": "mistral-embed",
		"input": inputs
	}
	
	for tries in range(4):
		response = requests.post(MISTRAL_API_EMBED_URL, json=data, headers=headers)
		if response.ok:
			break
		elif response.status_code == 429:
			wait_time = 2 ** tries
			#print(f"Waiting {wait_time} second(s)...")
			time.sleep(wait_time)
		else:
			print(response.text)
			response.raise_for_status()
	else:
		print(response.text)
		response.raise_for_status()
	
	
	embed_res = response.json()
	return [obj["embedding"] for obj in embed_res["data"]]
	

print(mistral_embed_texts(["Hello world"]))


	
class MistralLLM:
	
	def __init__(self, model="mistral-large-latest"):
		self.model = model
	
	def generate(
		self,
		prompt,
		return_json=False,
		**kwargs
	):
		if isinstance(prompt, str):
			prompt = [{"role":"user", "content":prompt}]
		
		format = "json_object" if return_json else "text"
			
		response = mistral_request(
			prompt,
			**kwargs,
			model=self.model,
			response_format={"type":format}
		)
		
		response = response["choices"][0]["message"]["content"]
		
		if return_json:
			return json_repair.loads(response)
			
		return response
	
