import os
import requests
import json_repair
import time
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
	
	max_delay = 20
	for tries in range(6):
		response = requests.post(MISTRAL_API_CHAT_URL, json=data, headers=headers)
		if response.ok:
			break
		elif response.status_code == 429:
			wait_time = min(max_delay, 2 ** (tries + 1))
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
	
	max_delay = 20
	for tries in range(4):
		response = requests.post(MISTRAL_API_EMBED_URL, json=data, headers=headers)
		if response.ok:
			break
		elif response.status_code == 429:
			wait_time = min(max_delay, 2 ** (tries + 1))
			#print(f"Waiting {wait_time} second(s)...")
			time.sleep(wait_time)
		else:
			print(response.text)
			response.raise_for_status()
	else:
		print(response.text)
		response.raise_for_status()
	
	embed_res = response.json()
	if isinstance(inputs, str):
		return embed_res["data"][0]["embedding"]
	return [obj["embedding"] for obj in embed_res["data"]]


def _convert_system_to_user(messages):
	new_messages = []
	for msg in messages:
		role = msg["role"]
		content = msg["content"]
		if role == "system":
			role = "user"
			content = f"<SYSTEM>{content}</SYSTEM>"
		new_messages.append({"role":role, "content":content})
	return new_messages


class MistralLLM:
	
	def __init__(self, model="mistral-large-latest"):
		self.model = model
		
	def generate(
		self,
		prompt,
		return_json=False,
		schema=None,
		**kwargs
	):
		if schema and not return_json:
			raise ValueError("return_json must be True if schema is provided")
		if isinstance(prompt, str):
			prompt = [{"role":"user", "content":prompt}]		
		if self.model not in ["mistral-small-latest", "mistral-large-latest"]:
			prompt = _convert_system_to_user(prompt)
		
		if schema:
			format = {
				"type":"json_schema",
				"json_schema":{
					"name": "json_format",
					"schema": schema,
					"strict": True
				}
			}
		else:
			format = {"type":"json_object"} if return_json else {"type":"text"} 
		response = mistral_request(
			prompt,
			**kwargs,
			model=self.model,
			response_format=format
		)
		
		response = response["choices"][0]["message"]["content"]
		
		if return_json:
			return json_repair.loads(response)
			
		return response
	
