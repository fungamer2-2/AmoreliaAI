import time
import re
import random
import numpy as np
import uuid
import math
from collections import deque
from datetime import datetime

from llm import MistralLLM, mistral_embed_texts
from emotion_system import Emotion, EmotionSystem, PersonalitySystem

from const import *
from utils import get_approx_time_ago_str, num_to_str_sign
from rank_bm25 import BM25Okapi


class MessageBuffer:
	
	def __init__(self, max_messages):
		self.max_messages = max_messages
		self.messages = deque(maxlen=max_messages)
		self.system_prompt = ""

	def set_system_prompt(self, prompt):
		self.system_prompt = prompt.strip()

	def add_message(self, role, content):
		self.messages.append({"role": role, "content": content})
		
	def pop(self):
		msg = self.messages.pop()
		return msg

	def flush(self):
		self.messages.clear()
		
	def to_list(self, include_system_prompt=True):
		history = []
		if include_system_prompt and self.system_prompt:
			history.append({"role":"system", "content":self.system_prompt})
		history.extend(msg.copy() for msg in self.messages)
		return history


IMPORTANCE_PROMPT = """Given the following memory, rate the importance of the memory on a scale from 1 to 10.

For example, memories corresponding to a score of 1 may include:
- Chit-chat greetings
- Chat about the weather with no information of significance

Memories corresponding to a score of 10 may include:
- Getting married
- A loved one passing away
- Etc.

Return a JSON obiect in the format:
`{{"importance": integer}}`

Below is the memory you are to rate:
<memory>{memory}</memory>"""

def rate_importance(memory):
	model = MistralLLM("open-mistral-nemo")
	prompt = IMPORTANCE_PROMPT.format(memory=memory)
	data = model.generate(
		prompt,
		temperature=0.0,
		return_json=True
	)
	return data.get("importance", 3)
	

def normalize_text(text):			
	text = text.lower()
	for symbol in ".,:;!?":
		text = text.replace(symbol, " ")
	
	text = " ".join(text.split())
	text = text.replace("’", "'")
	
	contractions = {
		"here's": "here is",
		"there's": "there is",
		"can't": "cannot",
		"don't": "do not",
		"doesn't": "does not",
		"didn't": "did not",
		"isn't": "is not",
		"aren't": "are not",
		"wasn't": "was not",
		"hasn't": "has not",
		"hadn't": "had not",
		"shouldn't": "should not",	
		"won't": "will not",
		"i'm": "i am",
		"you're": "you are",
		"we're": "we are",
		"they're": "they are",
		"i've": "i have",
		"you've": "you have",
		"we've": "we have",
		"they've": "they have",
		"y'all": "you all",	
		"that's": "that is",
		"it's": "it is",
		"it'd": "it would",
		"i'll": "i will",
		"you'll": "you will",
		"he'll": "he will",
		"she'll": "she will",
		"we'll": "we will",
		"they'll": "they will",
		"gonna": "going to",
		"could've": "could have",
		"should've": "should have",
		"would've": "would have",
		"gimme": "give me",
		"gotta": "got to",
		"how's": "how is",
	}
	def _replacement(match):
		bound1 = match.group(1)
		txt = match.group(2)
		bound2 = match.group(3)
		return f"{bound1}{contractions[txt]}{bound2}"
	
	for c in contractions:
		text = re.sub(rf"(\b)({c})(\b)", _replacement, text)
	return text
	
	
class Memory:
	
	def __init__(self, content):
		now = datetime.now()
		self.timestamp = now
		self.last_accessed = now
		self.content = content
		self.embedding = None
		self.id = str(uuid.uuid4())
		self.strength = 0	

	def get_recency_factor(self):
		seconds = (datetime.now() - self.last_accessed).total_seconds()
		days = seconds / 86400
		return math.exp(-days / ((self.strength + 1) * MEMORY_DECAY_TIME_MULT))

	def reinforce(self): 
		self.strength += 0.5
		self.last_accessed = datetime.now()

	def format_memory(self):
		timedelta = datetime.now() - self.timestamp
		time_ago_str = get_approx_time_ago_str(timedelta)
		return f"<memory timestamp=\"{self.timestamp}\" time_ago=\"{time_ago_str}\">{self.content}</memory>"

	def encode(self, embedding=None):
		if self.embedding is None:
			self.embedding = embedding or mistral_embed_texts(self.content)
			self.embedding = np.array(self.embedding)


class LSHMemory:
	
	def __init__(self, nbits, embed_size):
		# Number of buckets = 2 ** nbits
		self.table = {}
		self.memory_ids = {}
		rng = np.random.default_rng(seed=42)
		self.rand = rng.normal(size=(embed_size, nbits))
		
	def _get_hash(self, vec):
		proj = np.dot(vec, self.rand)
		bits = (proj > 0).astype(int)
		hash_ind = 0
		for bit in bits:
			hash_ind <<= 1
			hash_ind |= bit
		return hash_ind
		
	def add_memory(self, memory):
		vec = memory.embedding
		hash_ind = self._get_hash(vec)
		self.table.setdefault(hash_ind, [])
		self.table[hash_ind].append(memory)
		self.memory_ids[memory.id] = (memory, hash_ind)
		
	def delete_memory(self, memory):
		if memory.id not in self.memory_ids:
			return
		_, hash_ind = self.memory_ids[memory.id]
		bucket = self.table[hash_ind]
		for i in range(len(bucket)):
			if bucket[i].id == memory.id:
				del bucket[i]
				break
		
	def retrieve(self, query, k, remove=False):
		query_vec = mistral_embed_texts(query)
		hash_ind = self._get_hash(query_vec)
		
		memories = self.table.get(hash_ind, [])
		if not memories:
			return []
		
		k = min(k, len(memories))	
		result_vecs = np.stack([mem.embedding for mem in memories])
		sim_vals = (query_vec @ result_vecs.T)
		sim_vals /= np.linalg.norm(query_vec) * np.linalg.norm(result_vecs, axis=1)
		
		recency_vals = np.array([mem.get_recency_factor() for mem in memories])
		
		scores = sim_values + recency_vals
		
		idx = np.argpartition(scores, -k)[-k:]
		idx = idx[np.argsort(scores[idx])[::-1]]
		retrieved = [memories[i] for i in idx]
		if remove:
			for mem in retrieved:
				self.delete_memory(mem)
		return retrieved
		
	def recall_random(self, remove=False):
		# Recall a random subset of memories
		recalled = []
		for hash_ind in self.table:
			if not self.table[hash_ind]:
				continue
			sample_size = min(3, len(self.table[hash_ind]))
			sample = random.sample(self.table[hash_ind], sample_size)
			recalled.extend(sample)
		
		if len(recalled) > 5:
			recalled = random.sample(recalled, 5)

		if remove:
			for mem in recalled:
				self.delete_memory(mem)
		
		return recalled


class ShortTermMemory:
	capacity = 20
	
	# TODO: Add better memory rehearsal mechanism

	def __init__(self):
		self.memories = deque()
		
	def add_memory(self, memory):
		self.memories.append(memory) 
		
	def move_to_end(self, memory):
		if memory in self.memories:
			self.memories.remove(memory)
			self.memories.append(memory)
		
	def add_memories(self, memories):
		for mem in memories:
			self.add_memory(mem)
		
	def flush_old_memories(self):
		old_memories = []
		while len(self.memories) > self.capacity:
			old_memories.append(self.memories.popleft())
		return old_memories
		
	def clear_memories(self):
		self.memories.clear()
		
	def get_memories(self):
		return list(self.memories)
		
	def reinforce(self, query):
		if not self.memories:
			return
		
		# Similar memories are more likely to be rehearsed
		corpus = [memory.content for memory in self.memories]
		tokenized_corpus = [normalize_text(text).split() for text in corpus]
		bm25 = BM25Okapi(tokenized_corpus)
		scores = bm25.get_scores(normalize_text(query).split())
		
		reinforced = []
		for mem, score in zip(self.memories, scores):
			if random.random() < score:
				reinforced.append((score, mem))
		reinforced.sort(key=lambda p: p[0])
		for _, mem in reinforced:
			mem.reinforce()
			self.move_to_end(mem)

		
class LongTermMemory:
	
	def __init__(self):
		self.lsh = LSHMemory(LSH_NUM_BITS, LSH_VEC_DIM)
	
	def retrieve(self, query, k, remove=False):
		return self.lsh.retrieve(query, k, remove=remove)
	
	def recall_random(self, remove=False):
		return self.lsh.recall_random(remove=remove)
		
	def add_memory(self, memory):
		memory.encode()
		self.lsh.add_memory(memory)
		
	def add_memories(self, memories):
		memory_texts = [mem.content for mem in memories]
		embeddings = mistral_embed_texts(memory_texts)
		for memory, embed in zip(memories, embeddings):
			memory.encode(embed)
			self.lsh.add_memory(memory)
			
	def get_memories(self):
		return self.lsh.get_memories()


class MemorySystem:
	
	def __init__(self):
		self.short_term = ShortTermMemory()
		self.long_term = LongTermMemory()
		self.last_memory = datetime.now() 
		
	def remember(self, content):
		self.last_memory = datetime.now()
		self.short_term.add_memory(Memory(content))
		
	def recall(self, query):
		self.short_term.reinforce(query)
		memories = self.long_term.retrieve(query, 3, remove=True)
		for mem in memories:
			mem.reinforce()
			self.short_term.add_memory(mem)
		return memories
		
	def tick(self):
		now = datetime.now() 
		old_memories = self.short_term.flush_old_memories()
		for memory in old_memories:
			self.long_term.add_memory(memory)
		timedelta = now - self.last_memory
		if timedelta.total_seconds() > 6 * 3600:
			# Consolidate memories after 6 hours of inactivity
			self.consolidate_memories()
			self.last_memory = now
		
	def consolidate_memories(self):
		print("Consolidating all memories...")
		memories = self.short_term.get_memories()
		self.long_term.add_memories(memories)
		self.short_term.clear_memories()
		
	def surface_random_thoughts(self):
		memories = self.long_term.recall_random(remove=True)
		for mem in memories:
			mem.reinforce()
		self.short_term.add_memories(memories)
		
	def get_short_term_memories(self):
		return self.short_term.get_memories()
		
	def retrieve_memories(self, messages):
		role_map = {
			"user": "User",
			"assistant": "AI"
		}
		messages = [msg for msg in messages if msg["role"] != "system"]
		context = "\n".join(
			f"{role_map[msg['role']]}: {msg['content']}"
			for msg in messages[-3:]  # Use the last few messages as context		
		)
		short_term_memories = self.short_term.get_memories()
		long_term_memories = self.recall(context)
		return short_term_memories, long_term_memories
	

class ThoughtSystem:
	
	def __init__(
		self,
		emotion_system,
		memory_system
	):
		self.model = MistralLLM("mistral-large-latest")
		self.emotion_system = emotion_system
		self.memory_system = memory_system
		self.show_thoughts = True
		
	def think(
		self,
		messages,
		short_term_memories,
		long_term_memories
	):
		role_map = {
			"user": "User",
			"assistant": "AI"
		}
		history_str = "\n\n".join(
			f"{role_map[msg['role']]}: {msg['content']}"
			for msg in messages[:-1]
		)
		mood_prompt = self.emotion_system.get_mood_prompt()
		mood = self.emotion_system.mood
		
		short_term = "\n".join(mem.format_memory() for mem in short_term_memories)
		long_term = "\n".join(mem.format_memory() for mem in long_term_memories)
		
		prompt = THOUGHT_PROMPT.format(
			history_str=history_str,
			user_input=messages[-1]["content"],
			mood_long_desc=self.emotion_system.get_mood_long_description(),
			curr_date=datetime.now().strftime("%a, %-m/%-d/%Y"),
			curr_time=datetime.now().strftime("%-I:%M %p"),
			mood_prompt=mood_prompt,
			short_term=short_term,
			long_term=long_term
		)
		
		data = self.model.generate(
			[
				{"role":"system", "content":AI_SYSTEM_PROMPT},
				{"role":"user", "content":prompt}
			],
			temperature=0.7,
			return_json=True
		)
		intensity = int(data.get("emotion_intensity", 5))
		emotion = data["emotion"]
		
		if emotion not in EMOTION_MAP:
			for em in EMOTION_MAP:
				if em.lower() == emotion.lower():
					data["emotion"] = em
					break
		
		data["emotion_intensity"] = intensity
		
		self.emotion_system.experience_emotion(
			Emotion(*EMOTION_MAP[data["emotion"]]),
			intensity/10
		)
		
		if self.show_thoughts:
			print("AI thoughts:")
			for thought in data["thoughts"]:
				print(f"- {thought}")
			print()
		
		print(f"Emotion: {data['emotion']}, intensity {intensity}/10")
		print(f"Emotion reason: {data['emotion_reason']}")
		
		if data["insights"]:
			# Add new insights gained into memory
			print("Insights gained:")
			for insight in data["insights"]:
				print(f"- {insight}")
				self.memory_system.remember(f"I gained an insight while chatting with the user: {insight}")
		return data


class AISystem:

	def __init__(self):
		self.model = MistralLLM("mistral-large-latest")
		self.personality_system = PersonalitySystem(
			open=0.45,
			conscientious=0.25,
			extrovert=0.18,
			agreeable=0.93,
			neurotic=-0.15
		)
		
		self.memory_system = MemorySystem()
		self.emotion_system = EmotionSystem(self.personality_system)
		self.thought_system = ThoughtSystem(self.emotion_system, self.memory_system)
		
		self.last_message = datetime.now()
		self.last_login = None
		
		self.buffer = MessageBuffer(20)
		self.buffer.set_system_prompt(self.get_system_prompt())
		
	def get_system_prompt(self):
		prompt = AI_SYSTEM_PROMPT + "\n\nYour Personality Description: " + self.personality_system.get_summary()
		return prompt
		
	def get_mood(self):
		return self.emotion_system.mood
		
	def set_thoughts_shown(self, visible):
		self.thought_system.show_thoughts = visible
		
	def on_startup(self):
		self.buffer.flush()
		if self.last_login is None:
			self.buffer.add_message("user", "[The user has logged in for the first time]")
		self.last_login = datetime.now()	
		
	def send_message(self, user_input):
		self.tick()
		self.last_message = datetime.now()
		self.buffer.set_system_prompt(self.get_system_prompt())

		self.buffer.add_message("user", user_input)
		
		history = self.buffer.to_list()
		
		mood = self.get_mood()
		
		short_term_memories, long_term_memories = self.memory_system.retrieve_memories(history)
		short_term = "\n".join(mem.format_memory() for mem in short_term_memories)
		long_term = "\n".join(mem.format_memory() for mem in long_term_memories)
		print("Short term:")
		print(short_term)
		print()
		print("Long term:")
		print(long_term)
		print()
		thought_data = self.thought_system.think(
			self.buffer.to_list(False),
			short_term_memories,
			long_term_memories
		)
		
		history[-1]["content"] = USER_TEMPLATE.format(
			user_input=history[-1]["content"],
			ai_thoughts="\n".join("- " + thought for thought in thought_data["thoughts"]),
			emotion=thought_data["emotion"],
			emotion_reason=thought_data["emotion_reason"],
			curr_date=datetime.now().strftime("%a, %-m/%-d/%Y"),
			curr_time=datetime.now().strftime("%-I:%M %p"),
			short_term=short_term,
			long_term=long_term
		)
		response = self.model.generate(
			history,
			temperature=0.7
		)
		self.memory_system.remember(f"User: {user_input}\n\nAI: {response}")
		self.tick()
		self.emotion_system.print_mood()
		print()
		self.buffer.add_message("assistant", response)		
		return response

	def tick(self):
		self.emotion_system.tick()
		self.memory_system.tick()
		if (datetime.now() - self.last_message).total_seconds() > 2 * 3600:
			self.memory_system.surface_random_thoughts()
			print("Random thoughts surfaced")
		
	def save(self, path):
		import pickle
		with open(path, "wb") as file:
			pickle.dump(self, file)
	
	@staticmethod		
	def load(path):
		import pickle, os
		if os.path.exists(path):
			with open(path, "rb") as file:
				return pickle.load(file)


def _try_convert_arg(arg):
	try:
		return int(arg)
	except ValueError:
		pass
	
	try:
		return float(arg)
	except ValueError:
		pass
		
	return arg			


def _parse_args(arg_list_str):
	i = 0
	tokens = []
	last_tok = ""
	in_str = False
	escape = False
	while i < len(arg_list_str):
		char = arg_list_str[i]
		if not escape and char == '"':
			in_str = not in_str
			if not in_str:
				tokens.append(last_tok)
				last_tok = ""
		elif in_str:
			last_tok += char
		elif char == " ":
			if last_tok:		
				tokens.append(_try_convert_arg(last_tok))
				last_tok = ""
		else:
			last_tok += char
		i += 1
	if last_tok:
		tokens.append(_try_convert_arg(last_tok))
	return tokens
	

def command_parse(string):
	split = string.split(None, 1)
	if len(split) == 2:
		command, remaining = split
	else:
		command, remaining = string, ""
	args = remaining.split()
	return command, _parse_args(args)


PATH = "ai_system_save.pkl"

ai = AISystem.load(PATH)
if ai:
	print("AI loaded.")
else:
	ai = AISystem()

ai.on_startup()
while True:
	ai.tick()
	
	ai.emotion_system.print_mood()
	msg = input("User: ").strip()
	if not msg:
		continue
		
	if msg.startswith("/"):
		command, args = command_parse(msg[1:])
		if command == "set_pleasure" and len(args) == 1:
			value = args[0]
			if not isinstance(value, float):
				continue
			ai.emotion_system.set_emotion(pleasure=value)
		if command == "set_arousal" and len(args) == 1:
			value = args[0]
			if not isinstance(value, float):
				continue
			ai.emotion_system.set_emotion(arousal=value)
		elif command == "set_dominance" and len(args) == 1:
			value = args[0]
			if not isinstance(value, float):
				continue
			ai.emotion_system.set_emotion(dominance=value)
		elif command == "reset_mood":
			ai.emotion_system.reset_mood()	
		elif command == "consolidate_memories":
			ai.memory_system.consolidate_memories()
			
		#ai.save(PATH)
		continue
	
	print()
	print("AI: " + ai.send_message(msg))
	ai.save(PATH)