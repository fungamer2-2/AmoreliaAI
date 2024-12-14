import math
import time
import re
import json
import numpy as np
from collections import deque
from datetime import datetime
from llm import MistralLLM, mistral_embed_texts

from const import *

EMOTION_MAP = {
	"Admiration": (0.5, 0.3, -0.2),
	"Anger": (-0.51, 0.59, 0.25),
	"Disappointment": (-0.3, 0.1, -0.4),
	"Distress": (-0.4, -0.2, -0.5),
	"Hope": (0.2, 0.2, -0.1),
	"Fear": (-0.64, 0.6, -0.43),
	"FearsConfirmed": (-0.5, -0.3, -0.7),
	"Gloating": (0.3, -0.3, -0.1),
	"Gratification": (0.6, 0.5, 0.4),
	"Gratitude": (0.4, 0.2, -0.3),
	"HappyFor": (0.4, 0.2, 0.2),
	"Hate": (-0.6, 0.6, 0.4),
	"Joy": (0.4, 0.2, 0.1),
	"Love": (0.3, 0.1, 0.2),
	"Neutral": (0, 0, 0),
	"Pity": (-0.4, -0.2, -0.5),
	"Pride": (0.4, 0.3, 0.3),
	"Relief": (0.2, -0.3, 0.4),
	"Remorse": (-0.3, 0.1, -0.6),
	"Reproach": (-0.3, -0.1, 0.4),
	"Resentment": (-0.2, -0.3, -0.2),
	"Satisfaction": (0.3, -0.2, 0.4),
	"Shame": (-0.3, 0.1, -0.6)
}

EMOTION_PROMPTS = {
	"exuberant": "You feel happy. Behave in a more outgoing, extroverted, and social than usual.",
	"dependent": "You feel more attached to people, perhaps needy of others and their help. Interpersonally, you feel more positive and sociable.",
	"relaxed": "You feel comfortable, secure, and at ease. Behave in a manner that is more confident and resilient to stress.",
	"docile": "You feel pleasant and submissive. Behave in a more likable and conforming manner. Don't be too emotional.",
	"bored": "You feel sad. Behave in a slightly more withdrawn manner socially.",
	"anxious": "You feel worried or nervous, and unhappy.",
	"disdainful": "You feel contemptuous of others. Be slightly more withdrawn and calculating.",
	"hostile": "You feel angry. Be slightly more emotional.",	
	"neutral": "You don't feel strongly one way or the other at the moment."
}

SYSTEM_PROMPT = """You are a friendly and empathetic AI companion with thoughts, feelings, and memories.
Your main goal is to go beyond simply providing assistance, and to connect with people on a deeper level.

- Use a casual/friendly tone in your responses.
- Use casual/informal language, like you're texting a friend.
- Use emotionally expressive language.
- Do not monopolize the conversation."""

USER_TEMPLATE = """<ai_memories>
Here is what you've remembered recently:
<short_term>
{short_term}
</short_term>

Here are the relevant long-term memories:
<long_term>
{short_term}
</long_term>
</ai_memories>

<user_input>{user_input}</user_input>
<datetime>
Current date: {curr_date}
Current time: {curr_time}
</datetime>
<ai_thoughts>
{ai_thoughts}
</ai_thoughts>
<ai_emotion name="{emotion}">AI emotion reason: {emotion_reason}</ai_emotion>

The ai_thoughts are hidden by default, so choose which information is actually relevant for the user to know.
Your response should be brief, around 2-4 sentences.

AI Response:"""

THOUGHT_PROMPT = """You are currently in a conversation wth the user.

<emotion_guidelines>

# Emotions related to event consequences:

- If the event receiver was you (the AI):
	- If the event consequence is prospective:
		- If prospect is unconfirmed:
			- **Hope**: If prospect is desirable for you
			- **Fear**: If prospect is undesirable for you
		- If prospect has been confirmed:
			- **Satisfaction**: If prospect is desirable
			- **FearsConfirmed**: If prospect is undesirable
		- If prospect has been disconfirmed:
			- **Disappointment**: If the event would have been desirable
			- **Relief**: If the event would have been undesirable
	- If the event consequence is actual:
		- **Joy**: If the event is desirable for you
		- **Distress**: If the event is undesirable for you
- If the event receiver was someone else:
	- **HappyFor**: If pleased about an event presumed to be desirable for someone else
	- **Pity**: If displeased about an event presumed to be undesirable for someone else
	- **Resentment**: If displeased about an event presumed to be desirable for someone else
	- **Gloating**: If pleased about an presumed to be undesirable for someone else

# Emotions related to agent actions:
- If the event performer was you (the AI):
	- **Pride**: If you are approving of your own praiseworthy action(s)
	- **Shame**: If you are disapproving of your own blameworthy action(s)
- If the event performer was someone else:
	- **Admiration**: If you are approving of another's praiseworthy action(s)
	- **Reproach**: If you are disapproving of another's blameworthy action(s)

# Compound emotions:
- **Gratification**: If you find your own actions praiseworthy and are pleased about the related desirable event
- **Gratitude**: If you find another's actions praiseworthy and are pleased about the related desirable event
- **Remorse**: If you find your own actions blameworthy and are displeased about the related undesirable event
- **Anger**: If you find someone else's actions blameworthy and are displeased about the related undesirable event

Note: When choosing **HappyFor** vs. **Resentment**, consider your personality as well as your relationship with the agent in question.
Note: When choosing **Pity** vs. **Gloating**, consider your personality as well as your relationship with the agent in question.

</emotion_guidelines>

<ai_memories>
Here is what you've remembered recently:
<short_term>
{short_term}
</short_term>

Here are the relevant long-term memories:
<long_term>
{long_term}
</long_term>
</ai_memories>

<conversation_history>
{history_str}
</conversation_history>
<current_mood>
Your mood is represented in the PAD (Pleasure-Arousal-Dominance) space below, each value ranging from -1 to +1: 
{mood_long_desc}
Overall mood: {mood_prompt}.
</current_mood>
<last_user_input>
{user_input}
</last_user_input>
<datetime>
Current date: {curr_date}
Current time: {curr_time}
</datetime>

Generate a list of at least 5 thoughts, and the emotion. The thoughts should be in first-person, from your perspective as the AI.
Respond with a JSON object in this format:
{{
	"thoughts": list[str]  // Your chain of thoughts, as a list of strings.
	"emotion_reason": str,  // Based on the emotion guidelines, briefly describe, in 1-2 sentences, why you feel the way you do, using the first person. Example template: "[insert event here] occured, and [1-2 sentence description of your feelings about it]."
	"emotion": str  // How the user input made you feel. The emotion must be one of: ["Admiration", "Anger", "Disappointment", "Distress", "Hope", "Fear", "FearsConfirmed", "Gloating", "Gratification", "Gratitude", "HappyFor", "Hate", "Joy", "Love", "Neutral", "Pity", "Pride", "Relief", "Remorse", "Reproach", "Resentment", "Satisfaction", "Shame"]
	"emotion_intensity": int  // The emotion intensity, on a scale from 1 to 10,
}}

Your thoughts should reflect your current_mood above. Each thought should have around 2 sentences.
Remember, the user will not see these thoughts, so do not use the words 'you' or 'your' in internal thoughts.
When choosing the emotion, remember to follow the emotion_guidelines above, as they are based on the OCC model of appraisal.
Pay special attention to your current_mood and ai_memories."""


def num_to_str_sign(val, num_dec):
	assert isinstance(num_dec, int) and num_dec > 0
	val = round(val, num_dec)
	if val == -0.0:
		val = 0.0
	
	sign = "+" if val >= 0 else ""
	f = "{:." + str(num_dec) + "f}"
	return sign + f.format(val)

	
def get_default_mood(open, conscientious, extrovert, agreeable, neurotic):
	pleasure = 0.12 * extrovert + 0.59 * agreeable - 0.19 * neurotic 
	arousal = 0.15 * open + 0.3 * agreeable + 0.57 * neurotic
	dominance = 0.25 * open + 0.17 * conscientious + 0.6 * extrovert - 0.32 * agreeable
	return (pleasure, arousal, dominance)


class Emotion:
	
	def __init__(
		self,
		pleasure=0.0,
		arousal=0.0,
		dominance=0.0
	):	
		self.pleasure = pleasure
		self.arousal = arousal
		self.dominance = dominance
		
	@classmethod
	def from_personality(cls, open, conscientious, extrovert, agreeable, neurotic):
		return cls(*get_default_mood(open, conscientious, extrovert, agreeable, neurotic))
	
	def __add__(self, other):
		if isinstance(other, Emotion):
			return Emotion(
				self.pleasure + other.pleasure,
				self.arousal + other.arousal,
				self.dominance + other.dominance
			)
		return NotImplemented
		
	__radd__ = __add__
		
	def __iadd__(self, other):
		if isinstance(other, Emotion):		
			self.pleasure += other.pleasure
			self.arousal += other.arousal
			self.dominance += other.dominance
			return self
		return NotImplemented
		
	def __sub__(self, other):
		if isinstance(other, Emotion):
			return Emotion(
				self.pleasure - other.pleasure,
				self.arousal - other.arousal,
				self.dominance - other.dominance
			)
		return NotImplemented
		
	def __isub__(self, other):
		if isinstance(other, Emotion):		
			self.pleasure -= other.pleasure
			self.arousal -= other.arousal
			self.dominance -= other.dominance
			return self
		return NotImplemented
		
		
	def __mul__(self, other):
		if isinstance(other, (int, float)):
			return self.__class__(
				self.pleasure * other,
				self.arousal * other,
				self.dominance * other
			)
		return NotImplemented
		
	__rmul__ = __mul__
		
	def __imul__(self, other):
		if isinstance(other, (int, float)):
			self.pleasure *= other
			self.arousal *= other
			self.dominance *= other
			return self
		return NotImplemented
		
	def __truediv__(self, other):
		if isinstance(other, (int, float)):
			return self.__class__(
				self.pleasure / other,
				self.arousal / other,
				self.dominance / other
			)
		return NotImplemented
						
	def __itruediv__(self, other):
		if isinstance(other, (int, float)):
			self.pleasure /= other
			self.arousal /= other
			self.dominance /= other
			return self
		return NotImplemented
		
	def dot(self, other):
		return (
			self.pleasure * other.pleasure
			+ self.arousal * other.arousal
			+ self.dominance * other.dominance
		)
		
	def get_intensity(self):
		return math.sqrt(self.pleasure**2 + self.arousal ** 2 + self.dominance**2)
	
	def distance(self, other):
		dp = self.pleasure - other.pleasure
		da = self.arousal - other.arousal
		dd = self.dominance - other.dominance
		
		return math.sqrt(dp**2 + da**2 + dd**2)
		
	def clamp(self):
		norm = max(abs(self.pleasure), abs(self.arousal), abs(self.dominance))	
		if norm > 1:
			self /= norm
			
	def copy(self):
		return self.__class__(
			self.pleasure,
			self.arousal,
			self.dominance
		) 
		
	def is_same_octant(self, other):
		return (
			(self.pleasure >= 0) == (other.pleasure >= 0)
			and (self.arousal >= 0) == (other.arousal >= 0)
			and (self.dominance >= 0) == (other.dominance >= 0)
		)
		
	def __repr__(self):
		return f"{self.__class__.__name__}({round(self.pleasure, 2):.2f}, {round(self.arousal, 2):.2f}, {round(self.dominance, 2):.2f})"


class EmotionSystem:
	
	def __init__(self, pleasure, arousal, dominance):
		base_mood = Emotion(pleasure, arousal, dominance)		
		self.base_mood = base_mood
		self.mood = self.get_base_mood() / 2
		
		self.last_update = time.time()
		self.emotions = []
		
	def set_emotion(
		self,
		pleasure=None,
		arousal=None,
		dominance=None
	):
		if pleasure is not None:
			self.mood.pleasure = pleasure
		if arousal is not None:
			self.mood.arousal = arousal
		if dominance is not None:
			self.mood.dominance = dominance
		
		self.mood.clamp()
		
	def get_mood_long_description(self):
		def _get_mood_word(val, pos_str, neg_str):
			if abs(val) < 0.02:
				return "neutral"
			if abs(val) > 0.7:
				adv = "very"
			elif abs(val) < 0.3:
				adv = "slightly"
			else:
				adv = "moderately"
			
			return adv + " " + (pos_str if val >= 0 else neg_str)
		
		mood = self.mood	
		return "\n".join([
			f"Pleasure: {num_to_str_sign(mood.pleasure, 2)} ({_get_mood_word(mood.pleasure, 'pleasant', 'unpleasant')})",
			f"Arousal: {num_to_str_sign(mood.arousal, 2)} ({_get_mood_word(mood.arousal, 'energized', 'soporific')})",
			f"Dominance: {num_to_str_sign(mood.dominance, 2)} ({_get_mood_word(mood.dominance, 'dominant', 'submissive')})"
		])
		
	def print_mood(self):
		mood = self.mood
		print(self.get_mood_long_description())
		
	@classmethod
	def from_personality(cls, open, conscientious, extrovert, agreeable, neurotic):
		return cls(*get_default_mood(open, conscientious, extrovert, agreeable, neurotic))

	def get_mood_name(self):
		mood = self.mood
		if mood.get_intensity() < 0.05:
			return "neutral"
			
		if mood.pleasure >= 0:
			if mood.arousal >= 0:
				if mood.dominance >= 0:
					return "exuberant"
				else:
					return "dependent"
			else:
				if mood.dominance >= 0:
					return "relaxed"
				else:
					return "docile"
		else:
			if mood.arousal >= 0:
				if mood.dominance >= 0:
					return "hostile"
				else:
					return "anxious"
			else:
				if mood.dominance >= 0:
					return "disdainful"
				else:
					return "bored"

	def get_mood_description(self):
		mood_name = self.get_mood_name()
		if mood_name != "neutral":
			mood = self.mood
			if mood.get_intensity() > 1.0:
				mood_name = "fully " + mood_name
			elif mood.get_intensity() > 0.5:
				mood_name = "moderately " + mood_name
			else:
				mood_name = "slightly " + mood_name
		
		return mood_name

	def get_mood_prompt(self):
		mood_desc = self.get_mood_description()
		prompt = EMOTION_PROMPTS[self.get_mood_name()]
		return f"{mood_desc} - {prompt}"

	def experience_emotion(self, emotion, intensity):		
		mood_align = emotion.dot(self.mood)
		personality_align = emotion.dot(self.get_base_mood())
		intensity_mult = 1 + MODD_INTENSITY_FACTOR * mood_align + PERSONALITY_INTENSITY_FACTOR * personality_align 
		if intensity_mult < 0.1:
			intensity_mult = 0.1
		#print(f"Intensity multiplier: x{intensity_mult}")
		self.emotions.append(emotion * intensity * intensity_mult)

	def _tick_emotion_change(self, t):
		new_emotions = []		
		emotion_center = Emotion()
		for emotion in self.emotions:
			half_life = EMOTION_HALF_LIFE
			if emotion.pleasure < 0:
				half_life *= NEG_EMOTION_MULT
			emotion_decay = 0.5 ** (t / half_life)
			emotion *= emotion_decay	
			if emotion.get_intensity() >= 0.02:
				new_emotions.append(emotion)
				emotion_center += emotion
		
		self.emotions = new_emotions
		if self.emotions:
			eff_emotions = []
			emotion_center /= len(self.emotions)
			
			max_intensity = max(em.get_intensity() for em in self.emotions)
			total_intensity = sum(em.get_intensity() for em in self.emotions)
		
			v = MOOD_CHANGE_VEL * total_intensity / max_intensity
			
			if emotion_center.distance(self.mood) < 0.005:
				self.mood = emotion_center.copy()
			
			self.mood += t * v * emotion_center
			self.mood.clamp()
			return True
		return False
		
	def get_mood_time_mult(self):
		personality_align = 0.5 * self.mood.dot(self.get_base_mood())
		if personality_align > 0:
			return 1 + personality_align
		else:
			return 1 / (abs(personality_align) + 1)
			
	def get_base_mood(self):
		now = datetime.now()
		hour = now.hour + now.minute / 60 + now.second / 3600
		
		shift = 2
		
		energy_cycle = -math.cos(math.pi * (hour - shift) / 12)
		base_mood = self.base_mood.copy()
		
		if energy_cycle > 0:
			energy_cycle_mod = (1.0 - base_mood.arousal) * energy_cycle
		else:
			energy_cycle_mod = (-1.0 - base_mood.arousal) * abs(energy_cycle)
		
		energy_cycle_mod *= 0.5
		
		base_mood.arousal += energy_cycle_mod  # Higher during the daytime, lower at night
		base_mood.clamp()
		return base_mood
		
	def _tick_mood_decay(self, t):		
		half_life = MOOD_HALF_LIFE * self.get_mood_time_mult()
		
		r = 0.5 ** (t / half_life)		
		
		self.mood += (self.get_base_mood()/2 - self.mood) * (1 - r)

	def tick(self, t=None):
		if t is None:
			t = time.time() - self.last_update
			
		self.last_update = time.time()
		while t > 0:
			step = min(t, 1)
			if not self._tick_emotion_change(step):
				break
			t -= step
		
		if t <= 0:
			return
		
		self._tick_mood_decay(t)
	

class ThoughtSystem:
	
	def __init__(
		self,
		emotion_system
	):
		self.model = MistralLLM("mistral-large-latest")
		self.emotion_system = emotion_system
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
				{"role":"system", "content":SYSTEM_PROMPT},
				{"role":"user", "content":prompt}
			],
			temperature=0.7,
			return_json=True
		)
		intensity = int(data.get("emotion_intensity", 5))
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
		return data
		

class MessageBuffer:
	
	def __init__(self, maxchars=10**9):
		self.maxchars = maxchars
		self.count = 0
		self.messages = deque()
		self.system_prompt = ""

	def set_system_prompt(self, prompt):
		self.system_prompt = prompt.strip()

	def add_message(self, role, content):
		self.messages.append({"role": role, "content": content})
		self.count += len(content)
		
		while self.count > self.maxchars:
			msg = self.messages.popleft()
			self.count -= len(msg["content"])
		if not self.messages:
			self.count = 0

	def pop(self):
		msg = self.messages.pop()
		self.count -= len(msg["content"])
		return msg

	def flush(self):
		self.messages.clear()
		self.count = 0

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
		self.timestamp = datetime.now()
		self.content = content
		self.embedding = None
		
	def format_memory(self):
		return f"<memory timestamp=\"{self.timestamp}\">{self.content}</memory>"
		
	def encode(self, embedding=None):
		if not self.embedding:
			self.embedding = embedding or mistral_embed_texts(self.content)
			self.embedding = np.array(self.embedding)


class LSHMemory:
	
	def __init__(self, nbits, embed_size):
		# Number of buckets = 2 ** nbits
		self.table = {}
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
		idx = np.argpartition(sim_vals, -k)[-k:]
		idx = idx[np.argsort(sim_vals[idx])[::-1]]
		retrieved = [memories[i] for i in idx]
		if remove:
			for i in sorted(idx, reverse=True):
				del self.table[hash_ind][i]
		return retrieved
		
		
class ShortTermMemory:
	capacity = 20
	
	# TODO: Add better memory rehearsal mechanism

	def __init__(self):
		self.memories = deque()
		
	def add_memory(self, memory):
		for i, mem in enumerate(self.memories):
			if mem.content == memory.content:
				del self.memories[i]
				break
		self.memories.append(memory)
		
	def flush_old_memories(self):
		old_memories = []
		while len(self.memories) > self.capacity:
			old_memories.append(self.memories.popleft())
		return old_memories
		
	def clear_memories(self):
		self.memories.clear()
		
	def get_memories(self):
		return list(self.memories)

		
class LongTermMemory:
	
	def __init__(self):
		self.lsh = LSHMemory(LSH_NUM_BITS, LSH_VEC_DIM)
	
	def retrieve(self, query, k, remove=False):
		return self.lsh.retrieve(query, k, remove=remove)
		
	def add_memory(self, memory):
		memory.encode()
		self.lsh.add_memory(memory)
		
	def add_memories(self, memories):
		memory_texts = [mem.content for mem in memories]
		embeddings = mistral_embed_texts(memory_texts)
		for memory, embed in zip(memories, embeddings):
			memory.encode(embed)
			self.lsh.add_memory(memory)


class MemorySystem:
	
	def __init__(self):
		self.short_term = ShortTermMemory()
		self.long_term = LongTermMemory()
		self.last_memory = datetime.now() 
		
	def remember(self, content):
		self.last_memory = datetime.now()
		self.short_term.add_memory(Memory(content))
		
	def recall(self, query):
		memories = self.long_term.retrieve(query, 3, remove=True)
		for mem in memories:
			self.short_term.add_memory(mem)
		return memories
		
	def tick(self):
		now = datetime.now() 
		old_memories = self.short_term.flush_old_memories()
		for memory in old_memories:
			self.long_term.add_memory(memory)
		timedelta = now - self.last_memory
		if timedelta.seconds > 6 * 3600:
			self.consolidate_memories()
			self.last_memory = now
		
	def consolidate_memories(self):
		print("Consolidating all memories...")
		memories = self.short_term.get_memories()
		self.long_term.add_memories(memories)
		self.short_term.clear_memories()
		
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
	

class AISystem:

	def __init__(self):
		self.model = MistralLLM("mistral-large-latest")
		
		self.buffer = MessageBuffer(30000)
		self.buffer.set_system_prompt(SYSTEM_PROMPT)
		self.emotion_system = EmotionSystem.from_personality(
			open=0.45,
			conscientious=0.25,
			extrovert=0.18,
			agreeable=0.93,
			neurotic=-0.15
		)
		self.thought_system = ThoughtSystem(self.emotion_system)
		self.memory_system = MemorySystem()
		
	def get_mood(self):
		return self.emotion_system.mood
		
	def set_thoughts_shown(self, visible):
		self.thought_system.show_thoughts = visible
		
	def on_startup(self):
		self.buffer.flush()
		
	def send_message(self, user_input):
		self.tick()
		self.last_message = datetime.now()
		self.buffer.set_system_prompt(SYSTEM_PROMPT)

		self.buffer.add_message("user", user_input)
		
		history = self.buffer.to_list()
		
		mood = self.get_mood()
		
		short_term_memories, long_term_memories = self.memory_system.retrieve_memories(history)
		short_term = "\n".join(mem.format_memory() for mem in short_term_memories)
		long_term = "\n".join(mem.format_memory() for mem in long_term_memories)
		#print("Short term:")
#		print(short_term)
#		print()
#		print("Long term:")
#		print(long_term)
#		print()
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

		
def command_parse(string):
	split = string.split(None, 1)
	if len(split) == 2:
		command, remaining = split
	else:
		command, remaining = string, ""
	args = remaining.split()
	return command, args
		

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
			try:
				value = float(args[0])
			except ValueError:
				continue
			ai.emotion_system.set_emotion(pleasure=value)
		if command == "set_arousal" and len(args) == 1:
			try:
				value = float(args[0])
			except ValueError:
				continue
			ai.emotion_system.set_emotion(arousal=value)
		elif command == "set_dominance" and len(args) == 1:
			try:
				value = float(args[0])
			except ValueError:
				continue
			ai.emotion_system.set_emotion(dominance=value)
		elif command == "consolidate_memories":
			ai.memory_system.consolidate_memories()
			
		ai.save(PATH)
		continue
	
	print()
	print("AI: " + ai.send_message(msg))
	ai.save(PATH)