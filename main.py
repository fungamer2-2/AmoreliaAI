import time
import re

from collections import deque
from datetime import datetime

from llm import MistralLLM
from emotion_system import (
	Emotion,
	EmotionSystem,
	PersonalitySystem
)
from memory_system import MemorySystem
from const import *


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

GENERATE_USER_RESPONSES_PROMPT = """# Task

Given the following conversation, please suggest 3 to 5 possible responses that the USER could respond to the last ASSISTANT message given the conversation context.

# Role descriptions

- **USER**: These are messages from the human
- **ASSISTANT**: These are responses from the AI model

# Format Instructions

Respond in JSON format:
```
{{
	"possible_responses": list[str]  // The list of responses that the USER might give, based on the conversation context
}}
```

# Conversation History

Here is the conversation history so far:

```
{conversation_history}
```

Possible USER responses:"""

def rate_importance(memory):
	model = MistralLLM("open-mistral-nemo")
	prompt = IMPORTANCE_PROMPT.format(memory=memory)
	data = model.generate(
		prompt,
		temperature=0.0,
		return_json=True
	)
	return data.get("importance", 3)
	
def suggest_responses(conversation):
	history_str = "\n\n".join(
		f"{msg['role'].upper()}: {msg['content']}"
		for msg in conversation
		if msg["role"] != "system"
	)
	model = MistralLLM("open-mistral-nemo")
	prompt = GENERATE_USER_RESPONSES_PROMPT.format(
		conversation_history=history_str
	)
	
	data = model.generate(
		prompt,
		temperature=0.7,
		presence_penalty=0.5,
		frequency_penalty=0.1,
		return_json=True
	)
	return data["possible_responses"]




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
			presence_penalty=0.7,
			return_json=True
		)
		intensity = int(data.get("emotion_intensity", 5))
		emotion = data.get("emotion", "")
		insights = data.get("insights", [])
		
		if emotion not in EMOTION_MAP:
			for em in EMOTION_MAP:
				if em.lower() == emotion.lower():
					data["emotion"] = em
					break
			
		data["emotion_intensity"] = intensity
		data.setdefault("emotion_reason", "I feel this way based on how the conversation has been going.")
		emotion_vector = EMOTION_MAP.get(emotion, (0.0, 0.0, 0.0))
		self.emotion_system.experience_emotion(
			Emotion(*emotion_vector),
			intensity/10
		)
		
		if self.show_thoughts:
			print("AI thoughts:")
			for thought in data["thoughts"]:
				print(f"- {thought}")
			print()
		
		if insights:
			# Add new insights gained into memory
			print("Insights gained:")
			for insight in insights:
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
		self.last_tick = datetime.now()
		
		self.buffer = MessageBuffer(20)
		self.buffer.set_system_prompt(self.get_system_prompt())
		
	def get_system_prompt(self):
		prompt = AI_SYSTEM_PROMPT + "\n\nYour Personality Description: " + self.personality_system.get_summary()
		return prompt
		
	def get_message_history(self, include_system_prompt=True):
		return self.buffer.to_list(include_system_prompt)
		
	def get_mood(self):
		return self.emotion_system.mood
		
	def set_thoughts_shown(self, visible):
		self.thought_system.show_thoughts = visible
		
	def on_startup(self):
		self.buffer.flush()
		#if self.last_login is None:
#			self.buffer.add_message("user", "[The user has logged in for the first time]")
		self.last_login = datetime.now()
		
		if not hasattr(self, "last_tick"):
			self.last_tick = datetime.now()	
		
	def send_message(self, user_input):
		self.tick()
		self.last_message = datetime.now()
		self.buffer.set_system_prompt(self.get_system_prompt())

		self.buffer.add_message("user", user_input)
		
		history = self.get_message_history()
		
		mood = self.get_mood()
		
		short_term_memories, long_term_memories = self.memory_system.retrieve_memories(history)
		short_term = "\n".join(mem.format_memory() for mem in short_term_memories)
		long_term = "\n".join(mem.format_memory() for mem in long_term_memories)
		#print("Short term:")
#		print("\n".join(mem.format_memory(debug=True) for mem in short_term_memories))
#		print()
#		print("Long term:")
#		print("\n".join(mem.format_memory(debug=True) for mem in long_term_memories))
#		print()
		thought_data = self.thought_system.think(
			self.get_message_history(False),
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
		now = datetime.now()
		dt = (now - self.last_tick).total_seconds()
		self.emotion_system.tick()
		self.memory_system.tick(dt)
		if (datetime.now() - self.last_message).total_seconds() > 2 * 3600:
			self.memory_system.surface_random_thoughts()
			print("Random thoughts surfaced")
		self.last_tick = now
		
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
		elif command == "suggest":
			history = ai.get_message_history(False)
			
			if history:
				print("Suggesting possible user responses...")
				possible_responses = suggest_responses(history)
				print("Possible responses:")
				for response in possible_responses:
					print("- " + response)
			else:
				print("You need to have sent at least one message before you can use this command")	
		#ai.save(PATH)
		continue
	
	print()
	message = ai.send_message(msg)
	ai.save(PATH)
	print("AI: " + message)
	