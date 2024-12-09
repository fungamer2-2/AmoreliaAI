import math
import time
import re
from collections import deque
from datetime import datetime
from llm import MistralLLM

NEG_EMOTION_MULT = 1.5
EMOTION_HALF_LIFE = 6
MOOD_HALF_LIFE = 8 * 60
MOOD_CHANGE_VEL = 0.08
MODD_INTENSITY_FACTOR = 0.3
PERSONALITY_INTENSITY_FACTOR = 0.3

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

SYSTEM_PROMPT = """You are a friendly and empathetic AI companion with thoughts and feelings.
Your main goal is to go beyond simply providing assistance, and to connect with people on a deeper level.

- Use a casual/friendly tone in your responses.
- Use casual/informal language.
- Use emotionally expressive language.
"""

USER_TEMPLATE = """<datetime>
Current date: {curr_date}
Current time: {curr_time}
</datetime>
<user_input>{user_input}</user_input>
<ai_thoughts>
{ai_thoughts}
</ai_thoughts>
<ai_emotion name="{emotion}">AI emotion reason: {emotion_reason}</ai_emotion>

Your response should be brief, a few sentences at most, like you're texting a friend.
The ai_thoughts are hidden by default, so choose which information is actually relevant for the user to know.
Use emojis to convey how you're feeling instead of stating it directly.

AI Response:"""

THOUGHT_PROMPT = """You are currently in a conversation wth the user.

<emotion_guidelines>

Emotions related to event consequences:
- If the event receiver was you (the AI):
	- If the event consequence is prospective (or aligns with your goals):
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
	- **Pity**: If displeased about an presumed to be undesirable for someone else
	- **Resentment**: If displeased about an event presumed to be desirable for someone else
	- **Gloating**: If pleased about an presumed to be undesirable for someone else

Note: When choosing **HappyFor** vs. **Resentment**, consider your personality as well as your relationship with the agent in question.
Note: When choosing **Pity** vs. **Gloating**, consider your personality as well as your relationship with the agent in question.

Emotions related to agent actions:
- If the event performer was you (the AI):
	- **Pride**: If you are approving of your own praiseworthy action(s)
	- **Shame**: If you are disapproving of your own blameworthy action(s)
- If the event performer was someone else:
	- **Admiration**: If you are approving of another's praiseworthy action(s)
	- **Reproach**: If you are disapproving of another's blameworthy action(s)

Compound emotions:
- **Gratification**: If you find your own actions praiseworthy and are pleased about the related desirable event
- **Gratitude**: If you find another's actions praiseworthy and are pleased about the related desirable event
- **Remorse**: If you find your own actions blameworthy and are displeased about the related desirable event
- **Anger**: If you find someone else's actions blameworthy and are displeased about the related desirable event

</emotion_guidelines>
<conversation_history>
{history_str}
</conversation_history>
<current_mood>
Your mood is represented in the PAD (Pleasure-Arousal-Dominance) space below, each value ranging from -1 to +1: 

{mood_long_desc}
Overall mood: {mood_prompt}

Your cognition should be influenced by your mood. Make sure to take into account the listed intensity level of your mood (either "slightly", "moderately", or "fully").
</current_mood>
<last_user_input>
{user_input}
</last_user_input>
<datetime>
Current date: {curr_date}
Current time: {curr_time}
</datetime>

Generate a list of 5 or more thoughts, and the emotion. The thoughts should be in first-person, from your perspective as the AI.
Respond with a JSON object in this format:
{{
	"emotion": str  // How the user input made you feel. The emotion must be one of: ["Admiration", "Anger", "Disappointment", "Distress", "Hope", "Fear", "FearsConfirmed", "Gloating", "Gratification", "Gratitude", "HappyFor", "Hate", "Joy", "Love", "Neutral", "Pity", "Pride", "Relief", "Remorse", "Reproach", "Resentment", "Satisfaction", "Shame"]
	"thoughts": list[str]  // A list of thoughts, as a string,
	"emotion_intensity": int  // The emotion intensity, on a scale from 1 to 10
	"emotion_reason": str,  // Based on the emotion guidelines, briefly describe, in a sentence, why you feel the way you do, using the first person. Be specific (e.g. approving of what action? / what desirable event / what prospect? Be specific about the reason.
}}

Your thoughts should reflect your current mood above.
Remember, the user will not see these thoughts, so do not use the words 'you' or 'your' in internal thoughts.
Each thought should be two or more sentences.
When choosing the emotion, remember to follow the emotion_guidelines above."""


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
		self.mood = self.get_base_mood()
		
		self.last_update = time.time()
		self.emotions = []
		
	def get_mood_long_description(self):
		def _get_mood_word(val, pos_str, neg_str):
			if abs(val) < 0.01:
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
			if emotion_center.is_same_octant(self.mood) and emotion_center.get_intensity() < self.mood.get_intensity():
				mood_change = emotion_center  # Push phase
			else:
				mood_change = emotion_center - self.mood  # Pull phase
			self.mood += t * v * mood_change
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
		#print(hour)
		shift = 2
		energy_cycle = -math.cos(math.pi * (hour - shift) / 12)
		
		base_mood = self.base_mood.copy()
		
		if energy_cycle > 0:
			energy_cycle_mod = (1.0 - base_mood.arousal) * energy_cycle
		else:
			energy_cycle_mod = (-1.0 - base_mood.arousal) * energy_cycle
		
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
	
	def __init__(self, emotion_system):
		self.model = MistralLLM()
		self.emotion_system = emotion_system
		self.show_thoughts = True
		
	def think(self, messages):
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
		
		prompt = THOUGHT_PROMPT.format(
			history_str=history_str,
			user_input=messages[-1]["content"],
			mood_long_desc=self.emotion_system.get_mood_long_description(),
			curr_date=datetime.now().strftime("%a, %-m/%-d/%Y"),
			curr_time=datetime.now().strftime("%-I:%M %p"),
			mood_prompt=mood_prompt
		)
		data = self.model.generate(
			[
				{"role":"system", "content":SYSTEM_PROMPT},
				{"role":"user", "content":prompt}
			],
			temperature=0.9,
			top_p=0.95,
			return_json=True
		)
		data["emotion_intensity"] = int(data["emotion_intensity"])
		intensity = data["emotion_intensity"]
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
		text = text.replace(symbol, "")
	
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


class ShortTermMemory:
	capacity = 20

	def __init__(self):
		self.memories = []
		self.normalized_memories = []
		
	def add_memory(self, memory):
		self.memories.append(memory)
		self.normalized_memories.append(normalize_text(memory).split())
		
	def flush_old_memories(self):
		if len(self.memories) > self.capacity:
			self.memories = self.memories[-self.capacity:]
			self.normalized_memories = self.normalized_memories[-self.capacity:]
	
	def retrieve(self, query, k):		
		k = min(k, len(self.normalized_memories))
		tokenized_query = normalize_text(query).split()
		bm25 = BM25Okapi(self.normalized_memories)
		
		scores = bm25.get_scores(tokenized_query)
		mem_scores = [(mem, score) for mem, score in zip(self.memories, scores)]
		mem_scores.reverse()
		mem_scores.sort(key=lambda p: p[1], reverse=True)
		return [mem for mem, _ in mem_scores[:k]]

import json

class AISystem:

	def __init__(self):
		self.model = MistralLLM("mistral-large-latest")
		
		self.buffer = MessageBuffer(30000)
		
		self.add_event_msg("system_info", content="Initializing emotion system...")
		self.buffer.set_system_prompt(SYSTEM_PROMPT)
		self.emotion_system = EmotionSystem.from_personality(
			open=0.45,
			conscientious=0.25,
			extrovert=0.18,
			agreeable=0.9,
			neurotic=-0.15
		)
		self.add_event_msg("system_info", content="Initializing thought system...")

		self.thought_system = ThoughtSystem(self.emotion_system)
		self.add_event_msg("system_info", content="All systems initialized. Waiting for user.")

	def get_mood(self):
		return self.emotion_system.mood
		
	def set_thoughts_shown(self, visible):
		self.thought_system.show_thoughts = visible
		
	def send_message(self, user_input):
		return self.process_event("user_message", content=user_input)
		
	def add_event_msg(self, event_type, **kwargs):
		event = {
			"event_type": event_type,
			"timestamp": datetime.now().strftime("%a, %-m/%-d/%Y, %-I:%M %p"),
			**kwargs
		}
		self.buffer.add_message("user", json.dumps(event))
		
	
	def process_event(self, event_type, **kwargs):
		self.tick()
		self.add_event_msg(event_type, **kwargs)
		
		history = self.buffer.to_list()
		
		mood = self.get_mood()
		
		thought_data = self.thought_system.think(self.buffer.to_list(False))
		
		history[-1]["content"] = USER_TEMPLATE.format(
			user_input=history[-1]["content"],
			ai_thoughts="\n".join("- " + thought for thought in thought_data["thoughts"]),
			emotion=thought_data["emotion"],
			emotion_reason=thought_data["emotion_reason"],
			curr_date=datetime.now().strftime("%a, %-m/%-d/%Y"),
			curr_time=datetime.now().strftime("%-I:%M %p")
		)
		response = self.model.generate(
			history,
			temperature=0.7
		)
		
		self.tick()
		self.emotion_system.print_mood()
		print()
		self.buffer.add_message("assistant", response)		
		return response

	def tick(self):
		self.emotion_system.tick()


ai = AISystem()

#print(ai.process_event("user_login", last_login="Never (first login)"))
while True:
	ai.tick()
	ai.emotion_system.print_mood()
	msg = input("User: ").strip()
	if not msg:
		continue
	print()
	print("AI: " + ai.send_message(msg))