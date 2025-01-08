import uuid
import numpy as np
import random
import math

from collections import deque
from datetime import datetime
from rank_bm25 import BM25Okapi

from const import * 
from llm import mistral_embed_texts
from utils import (
	normalize_text,
	get_approx_time_ago_str
)


class Memory:
		
	def __init__(self, content):
		now = datetime.now()
		self.timestamp = now
		self.last_accessed = now
		self.content = content
		self.embedding = None
		self.id = str(uuid.uuid4())
		self.strength = 1	

	def get_recency_factor(self):
		seconds = (datetime.now() - self.last_accessed).total_seconds()
		days = seconds / 86400
		return math.exp(-days / (self.strength * MEMORY_DECAY_TIME_MULT))
	
	def get_retention_prob(self):
		# The probability of retaining this memory per 24 hours
		recency = self.get_recency_factor()
		threshold = 0.6
		if recency > threshold:
			return 1.0
		return math.exp(-1 / (MEMORY_DECAY_TIME_MULT * self.strength))

	def reinforce(self): 
		self.strength += 1
		self.last_accessed = datetime.now()

	def format_memory(self, debug=False):
		timedelta = datetime.now() - self.timestamp
		time_ago_str = get_approx_time_ago_str(timedelta)
		if debug:
			strength = self.strength
			return f"<memory timestamp=\"{self.timestamp}\" time_ago=\"{time_ago_str}\" strength=\"{strength}\">{self.content}</memory>"

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
		
		scores = sim_vals + recency_vals
		
		idx = np.argpartition(scores, -k)[-k:]
		idx = idx[np.argsort(scores[idx])[::-1]]
		retrieved = [memories[i] for i in idx]
		if remove:
			for mem in retrieved:
				self.delete_memory(mem)
		return retrieved
		
	def get_memories(self):
		memories = []
		for bucket in self.table.values():
			memories.extend(bucket)
		return memories
	
	def recall_random(self, remove=False):
		# Recall a random subset of memories
		recalled = []
		weights = []
		for hash_ind in self.table:
			if not self.table[hash_ind]:
				continue
			sample_size = min(5, len(self.table[hash_ind]))
			sample = random.sample(self.table[hash_ind], sample_size)
			recalled.extend(sample)
			weights.extend([mem.strength for mem in sample_size])
		
		if len(recalled) > 5:
			new_recalled = []
			for _ in range(5):
				choice = random.choices(recalled, weights)[0]
				ind = recalled.index(choice)
				new_recalled.append(recalled.pop(ind))
				weights.pop(ind)
			
			recalled = new_recalled

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
		for mem in self.memories:
			if mem.content.lower() == memory.content.lower():
				self.move_to_end(mem)
				break
		else:
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
		
	def rehearse(self, query):
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
		reinforced = reinforced[-3:]
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
	
	def forget_memory(self, memory):
		self.lsh.delete_memory(memory)
		
	def tick(self, dt):
		for mem in self.get_memories():
			retain_prob = mem.get_retention_prob()
			if retain_prob >= 1.0:
				continue
			
			forget_prob = 1 - retain_prob	
			prob = 1 - ((1 - forget_prob) ** (dt / 86400))	
			if random.random() < prob:
				print("Forgot memory because it has not been recalled in a while.")
				print(f"Forgotten memory content: {mem.content}")
				self.forget_memory(mem)
	

class MemorySystem:
	
	def __init__(self, config):
		self.config = config
		self.short_term = ShortTermMemory()
		self.long_term = LongTermMemory()
		self.last_memory = datetime.now() 
		
	def remember(self, content):
		self.last_memory = datetime.now()
		self.short_term.add_memory(Memory(content))
		
	def recall(self, query):
		self.short_term.rehearse(query)
		memories = self.long_term.retrieve(query, 3, remove=True)
		for mem in memories:
			mem.reinforce()
			self.short_term.add_memory(mem)
		return memories
		
	def tick(self, dt):
		now = datetime.now() 
		old_memories = self.short_term.flush_old_memories()
		for memory in old_memories:
			self.long_term.add_memory(memory)
		timedelta = now - self.last_memory
		if timedelta.total_seconds() > 6 * 3600:
			# Consolidate memories after 6 hours of inactivity
			self.consolidate_memories()
			self.last_memory = now
		
		self.long_term.tick(dt)
		
	def consolidate_memories(self):
		print("Consolidating all memories...")
		memories = self.short_term.get_memories()
		for mem in memories:
			mem.strength += 1
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
			"assistant": self.config.name
		}
		messages = [msg for msg in messages if msg["role"] != "system"]
		context = "\n".join(
			f"{role_map[msg['role']]}: {msg['content']}"
			for msg in messages[-3:]  # Use the last few messages as context		
		)
		self.recall(context)
		return self.get_short_term_memories()
