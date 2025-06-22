import uuid
import random
import math
from collections import deque
from datetime import datetime

import numpy as np
from rank_bm25 import BM25Okapi

from const import * 
from llm import MistralLLM, mistral_embed_texts
from utils import (
	normalize_text,
	get_approx_time_ago_str,
	conversation_to_string
)
from emotion_system import Emotion
from belief_system import BeliefSystem


IMPORTANCE_PROMPT = """Your task is to rate the importance of the given memory from 1 to 10.

- A score of 1 represents trivial things or basic chit-chat with no information of importance.
- A score of 10 represents things that are very important.

Return ONLY an integer, nothing else. Your response must not contain any non-numeric characters.

<memory>
{memory}
</memory>

The importance score of the above memory is <fill_in_the_blank_here>/10.
"""


def get_importance(memory):
	"""Rates the importance of a given memory from 1-10."""
	model = MistralLLM("open-mistral-nemo")
	prompt = IMPORTANCE_PROMPT.format(
		memory=memory
	)
	output = model.generate(prompt, temperature=0.0)
	try:
		score = int(output)
	except ValueError:
		#print("Error parsing response to integer; returning default score 3/10.")
		score = 3
	
	return max(1, min(score, 10))
	
	
def cosine_similarity(x, y):
	x = np.array(x)
	y = np.array(y)
	assert x.ndim == 1 and y.ndim == 1
	x = x[np.newaxis, ...]
	y = y[np.newaxis, ...]
	sim = x @ y.T
	sim /= np.linalg.norm(x) * np.linalg.norm(y, axis=1)
	return np.squeeze(sim)



class Memory:
	"""Represents a stored memory"""
		
	def __init__(self, content, strength=1.0, emotion=None):
		now = datetime.now()
		self.timestamp = now
		self.last_accessed = now
		self.content = content
		self.embedding = None
		self.id = str(uuid.uuid4())
		self.strength = strength
		self.emotion = emotion or Emotion()

	def get_recency_factor(self, from_creation=False):
		"""Returns the recency value of a memory, based on time and strength"""
		t = self.timestamp if from_creation else self.last_accessed
		seconds = (datetime.now() - t).total_seconds()
		days = seconds / 86400
		return math.exp(-days / (self.strength * MEMORY_DECAY_TIME_MULT))

	def get_retention_prob(self):
		"""Calculates the probability of retaining this memory per 24 hours"""
		recency = self.get_recency_factor()
		if recency > MEMORY_RECENCY_FORGET_THRESHOLD:
			return 1.0
		return math.exp(-1 / (MEMORY_DECAY_TIME_MULT * self.strength))

	def reinforce(self):
		"""Reinforces the memory when it is recalled"""
		self.strength += 1
		self.last_accessed = datetime.now()

	def format_memory(self):
		"""Formats the memory as a string"""
		timedelta = datetime.now() - self.timestamp
		time_ago_str = get_approx_time_ago_str(timedelta)
		
		return f"<memory timestamp=\"{self.timestamp}\"" \
			f" time_ago=\"{time_ago_str}\">{self.content}</memory>"

	def encode(self, embedding=None):
		"""Generates a semantic embedding for the memory if it has not been created"""
		if self.embedding is None:
			self.embedding = embedding or mistral_embed_texts(self.content)
			self.embedding = np.array(self.embedding)


class LSHMemory:
	"""Stores long-term memories using locality-sensitive hashing"""
	
	def __init__(self, nbits, embed_size):
		# Number of buckets = 2 ** nbits
		self.table = {}
		self.memory_ids = {}
		rng = np.random.default_rng(seed=42)
		self.rand = rng.normal(size=(embed_size, nbits))
		self.count = 0
		
	def _get_hash(self, vec):
		proj = np.dot(vec, self.rand)
		bits = (proj > 0).astype(int)
		hash_ind = 0
		for bit in bits:
			hash_ind <<= 1
			hash_ind |= bit
		return hash_ind
		
	#def _cluster_memories(self, bucket):
#		threshold = 0.95
#		representatives = []
#		for memory in bucket:
#			embed = memory.embedding
#			if not representatives:
#				representatives.append(embed)
#			else:
#				similarity = -1.0
#				for repr_embed in representatives:
#					repr_sim = cosine_similarity(embed, repr_embed)
#					if repr_sim > similarity:
#						similarity = repr_sim
#				if similarity < threshold:
#					representatives.append(embed)
#
#		clusters = [[] for _ in range(len(representatives))]
#		for memory in bucket:
#			similarity = -1.0
#			idx = 0
#			for i, repr_embed in enumerate(representatives):
#				repr_sim = cosine_similarity(embed, repr_embed)
#				if repr_sim > similarity:
#					similarity = repr_sim
#					idx = i
#			clusters[idx].append(memory)
#		return clusters
#
	#def _prune_similar_memories(self, bucket):
#		# Prunes similar/less important memories
#		max_duplicates_allowed = 3
#		clusters = self._cluster_memories(bucket)
#
#		# Memories are more likely to be pruned based on:
#		# Higher similarity to other memories
#		for cluster in clusters:
#			cluster.sort(key=lambda m: m.get_recency_factor(), reverse=True)
#			to_remove = cluster[max_duplicates_allowed:]
#			for memory in to_remove:
#				self.delete_memory(memory)
#
	#def prune_memories(self):
#		for bucket in self.table.values():
#			self._prune_similar_memories(bucket)
#
	def add_memory(self, memory):
		"""Adds a memory"""
		self.count += 1
		vec = memory.embedding
		hash_ind = self._get_hash(vec)
		self.table.setdefault(hash_ind, [])
		self.table[hash_ind].append(memory)
		self.memory_ids[memory.id] = (memory, hash_ind)
	
	def delete_memory(self, memory):
		"""Removes a memory"""
		if memory.id not in self.memory_ids:
			return
		_, hash_ind = self.memory_ids[memory.id]
		bucket = self.table[hash_ind]
		for i, mem in enumerate(bucket):
			if mem.id == memory.id:
				del bucket[i]
				del self.memory_ids[memory.id]
				self.count -= 1
				break
	
	def retrieve(self, query, k, remove=False):
		"""Gets the top K most relevant memories"""
		if not self.count:
			return []
		query_vec = mistral_embed_texts(query)
		hash_ind = self._get_hash(query_vec)

		memories = self.table.get(hash_ind, [])
		if not memories:
			return []

		k = min(k, len(memories))
		result_vecs = np.stack([mem.embedding for mem in memories])
		sim_vals = query_vec @ result_vecs.T
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
		"""Gets all memories as a list"""
		memories = []
		for bucket in self.table.values():
			memories.extend(bucket)
		return memories
	
	def recall_random(self, remove=False):
		"""Recalls a random subset of memories, weighted by memory strength"""
		recalled = []
		weights = []
		for bucket in self.table.values():
			if not bucket:
				continue
			sample_size = min(6, len(bucket))
			sample = random.sample(bucket, sample_size)
			recalled.extend(sample)
			weights.extend([mem.get_recency_factor() for mem in sample])
	
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
	"""Short-term memory that stores recently accessed or experienced memories"""
	capacity = 20

	def __init__(self):
		self.memories = deque()

	def add_memory(self, memory):
		"""Adds a new memory"""
		for mem in self.memories:
			if mem.content.lower() == memory.content.lower():
				self._move_to_end(mem)
				break
		else:
			self.memories.append(memory)

	def _move_to_end(self, memory):
		if memory in self.memories:
			self.memories.remove(memory)
			self.memories.append(memory)

	def add_memories(self, memories):
		"""Adds a list of memories"""
		for mem in memories:
			self.add_memory(mem)

	def flush_old_memories(self):
		"""Flushes out and returns memories that have exceeded the capacity"""
		old_memories = []
		while len(self.memories) > self.capacity:
			old_memories.append(self.memories.popleft())
		return old_memories

	def clear_memories(self):
		"""Clears all short-term memories"""
		self.memories.clear()

	def get_memories(self):
		"""Returns a list of all short-term memories"""
		return list(self.memories)

	def rehearse(self, query):
		"""Strengthens any similar memories to the query"""
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
			self._move_to_end(mem)

		
class LongTermMemory:
	"""Long-term memory which stores memories long-term"""

	def __init__(self):
		self.lsh = LSHMemory(LSH_NUM_BITS, LSH_VEC_DIM)
	
	def retrieve(self, query, k, remove=False):
		"""Returns the top K most relevant memories"""
		return self.lsh.retrieve(query, k, remove=remove)

	def recall_random(self, remove=False):
		"""Recalls a random subset of memories"""
		return self.lsh.recall_random(remove=remove)

	def add_memory(self, memory):
		"""Adds a new long-term memory"""
		memory.encode()
		self.lsh.add_memory(memory)

	def add_memories(self, memories):
		"""Adds a list of long-term memories"""
		if not memories:
			return
		memory_texts = [mem.content for mem in memories]
		embeddings = mistral_embed_texts(memory_texts)
		for memory, embed in zip(memories, embeddings):
			memory.encode(embed)
			self.lsh.add_memory(memory)

	def get_memories(self):
		"""Returns a list of all long-term memories"""
		return self.lsh.get_memories()

	def forget_memory(self, memory):
		"""Removes a memory from long-term"""
		self.lsh.delete_memory(memory)

	def tick(self, delta):
		"""Runs an update tick"""
		for mem in self.get_memories():
			retain_prob = mem.get_retention_prob()
			if retain_prob >= 1.0:
				continue
			forget_prob = 1 - retain_prob
			prob = 1 - ((1 - forget_prob) ** (delta / 86400))
			if random.random() < prob:
				print("Forgot memory because it has not been recalled in a while.")
				print(f"Forgotten memory content: {mem.content}")
				self.forget_memory(mem)
	

class MemorySystem:
	"""The AI's memory system"""
	def __init__(self, config):
		self.config = config
		self.short_term = ShortTermMemory()
		self.long_term = LongTermMemory()
		self.last_memory = datetime.now()
		self.belief_system = BeliefSystem(config)
		self.importance_counter = 0.0
		
	def get_beliefs(self):
		return self.belief_system.get_beliefs()
	
	def reset_importance(self):
		"""Resets the importance counter"""
		self.importance_counter = 0.0
	
	def remember(self, content, emotion=None, is_insight=False):
		"""Adds a new memory"""
		importance = get_importance(content)
		strength = 1 + (importance - 1) / 2
		self.last_memory = datetime.now()
		self.short_term.add_memory(Memory(content, strength=strength, emotion=emotion))
		self.importance_counter += importance / 10
		#print(f"Importance: {importance}")
		if not is_insight and importance >= 6:  # Important memories will create new beliefs
			self.belief_system.generate_new_belief(content, importance/10)

	def recall(self, query):
		"""Recalls and returns the most relevant memories"""
		self.short_term.rehearse(query)
		memories = self.long_term.retrieve(query, MEMORY_RETRIEVAL_TOP_K, remove=True)
		for mem in memories:
			mem.reinforce()
			self.short_term.add_memory(mem)
		return memories
	
	def tick(self, dt):
		"""Runs an update tick"""
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
		self.belief_system.tick(dt)
		
	def consolidate_memories(self):
		"""Consilidates all short-term memories into long-term"""
		print("Consolidating all memories...")
		memories = self.short_term.get_memories()	
		self.long_term.add_memories(memories)
		self.short_term.clear_memories()
		
	def surface_random_thoughts(self):
		"""Brings a random subset of thoughts to short-term memory"""
		memories = self.long_term.recall_random(remove=True)
		for mem in memories:
			mem.reinforce()
		self.short_term.add_memories(memories)
		
	def get_short_term_memories(self):
		"""Gets a list of all short-term memories"""
		return self.short_term.get_memories()
		
	def retrieve_long_term(self, query, top_k):
		"""Retrieves the top K most relevant memories from long-term memory"""
		return self.long_term.retrieve(query, top_k, remove=False)
	
	def recall_memories(self, messages):
		"""Returns the short-term and recalled long-term memories given the query"""
		messages = [msg for msg in messages if msg["role"] != "system"]
		context = conversation_to_string(messages[-3:])
		recalled_memories = self.recall(context)
		return self.get_short_term_memories(), recalled_memories
