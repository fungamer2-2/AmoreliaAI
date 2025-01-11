import time
import re
import os
import traceback

from collections import deque
from datetime import datetime

from colored import Fore, Style	
from pydantic import BaseModel, Field

from llm import MistralLLM
from utils import clear_screen
from emotion_system import (
	Emotion,
	EmotionSystem,
	PersonalitySystem,
	RelationshipSystem
)
from memory_system import MemorySystem
from thought_system import ThoughtSystem
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


GENERATE_USER_RESPONSES_PROMPT = """# Task

Given the following conversation, please suggest 3 to 5 possible responses that the HUMAN could respond to the last AI message given the conversation context.

# Role descriptions

- **HUMAN**: These are messages from the human
- **AI**: These are responses from the AI model

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

Possible **HUMAN** responses:"""
	
def suggest_responses(conversation):
	role_map = {
		"user": "HUMAN",
		"assistant": "AI"
	}
	history_str = "\n\n".join(
		f"{role_map[msg['role']]}: {msg['content']}"
		for msg in conversation
		if msg["role"] != "system"
	)
	model = MistralLLM("mistral-small-latest")
	prompt = GENERATE_USER_RESPONSES_PROMPT.format(
		conversation_history=history_str
	)
	
	data = model.generate(
		prompt,
		temperature=0.7,
		return_json=True
	)
	return data["possible_responses"]


class PersonalityConfig(BaseModel):
	open: float = Field(ge=-1.0, le=1.0)
	conscientious: float = Field(ge=-1.0, le=1.0)
	agreeable: float = Field(ge=-1.0, le=1.0)
	extrovert: float = Field(ge=-1.0, le=1.0)
	neurotic: float = Field(ge=-1.0, le=1.0)
	

class AIConfig(BaseModel):
	name: str = Field(default="AI")
	system_prompt: str = Field(
		default=AI_SYSTEM_PROMPT
	)
	personality: PersonalityConfig = Field(
		default_factory=lambda: PersonalityConfig(
			open=0.45,
			conscientious=0.25,
			extrovert=0.18,
			agreeable=0.93,
			neurotic=-0.15
		)
	)


class AISystem:

	def __init__(self, config=None):
		config = config or AIConfig()
		
		personality = config.personality
		
		self.config = config
		self.model = MistralLLM("mistral-large-latest")
		self.name = config.name
		self.personality_system = PersonalitySystem(
			open=personality.open,
			conscientious=personality.conscientious,
			extrovert=personality.extrovert,
			agreeable=personality.agreeable,
			neurotic=personality.neurotic
		)
		self.memory_system = MemorySystem(config)
		self.relation_system = RelationshipSystem()
		self.emotion_system = EmotionSystem(
			self.personality_system,
			self.relation_system
		)
		self.thought_system = ThoughtSystem(
			config,
			self.emotion_system,
			self.memory_system,
			self.relation_system,
			self.personality_system
		)
		
		self.last_message = datetime.now()
		self.last_login = None
		self.last_tick = datetime.now()
		
		self.buffer = MessageBuffer(20)
		self.buffer.set_system_prompt(config.system_prompt)
		
	def get_message_history(self, include_system_prompt=True):
		return self.buffer.to_list(include_system_prompt)
		
	def get_mood(self):
		return self.emotion_system.mood
		
	def set_thoughts_shown(self, visible):
		self.thought_system.show_thoughts = visible
		
	def on_startup(self):
		self.buffer.flush()
		self.last_login = datetime.now()
		
		if not hasattr(self, "last_tick"):
			self.last_tick = datetime.now()
		self.tick()
		
	def send_message(self, user_input):
		self.tick()
		self.last_message = datetime.now()
		self.buffer.set_system_prompt(self.config.system_prompt)

		self.buffer.add_message("user", user_input)
		
		history = self.get_message_history()
		
		mood = self.get_mood()
		
		memories = self.memory_system.retrieve_memories(history)
		memories_str = (
			"\n".join(mem.format_memory() for mem in memories)
			if memories 
			else "You don't have any memories of this user yet!"
		)
		thought_data = self.thought_system.think(
			self.get_message_history(False),
			memories
		)
		user_emotions = thought_data["possible_user_emotions"]
		if user_emotions:
			user_emotion_str = "The user appears to be feeling the following emotions: " + ", ".join(user_emotions) + "."
		else:
			user_emotion_str = "The user doesn't appear to show any strong emotion."
		history[-1]["content"] = USER_TEMPLATE.format(
			name=self.config.name,
			user_input=history[-1]["content"],
			personality_summary=self.personality_system.get_summary(),
			ai_thoughts="\n".join("- " + thought for thought in thought_data["thoughts"]),
			emotion=thought_data["emotion"],
			emotion_reason=thought_data["emotion_reason"],
			emotion_influence=thought_data["emotion_influence"],
			curr_date=datetime.now().strftime("%a, %-m/%-d/%Y"),
			curr_time=datetime.now().strftime("%-I:%M %p"),
			memories=memories_str,
			user_emotion_str=user_emotion_str
		)
		response = self.model.generate(
			history,
			temperature=0.8,
			presence_penalty=0.6
		)
		self.memory_system.remember(f"User: {user_input}\n\n{self.name}: {response}")
		self.tick()
		self.buffer.add_message("assistant", response)		
		return response

	def tick(self):
		now = datetime.now()
		dt = (now - self.last_tick).total_seconds()
		self.emotion_system.tick()
		self.memory_system.tick(dt)
		if dt > 2 * 3600:
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



ai = AISystem.load(SAVE_PATH)
is_new = ai is None
if is_new:
	ai = AISystem()
	print("AI system initialized.")
else:
	print("AI loaded.")
	

ai.on_startup()
if not is_new:
	ai.save(SAVE_PATH)

print(f"{Fore.yellow}Note: It's recommended not to enter any sensitive information.{Style.reset}")

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
			if not isinstance(value, (int, float)):
				continue
			ai.emotion_system.set_emotion(pleasure=value)
		if command == "set_arousal" and len(args) == 1:
			value = args[0]
			if not isinstance(value, (int, float)):
				continue
			ai.emotion_system.set_emotion(arousal=value)
		elif command == "set_dominance" and len(args) == 1:
			value = args[0]
			if not isinstance(value, (int, float)):
				continue
			ai.emotion_system.set_emotion(dominance=value)
		elif command == "set_relation_friendliness" and len(args) == 1:
			value = args[0]
			if not isinstance(value, (int, float)):
				continue
			ai.relation_system.set_relation(friendliness=value)
		elif command == "set_relation_dominance" and len(args) == 1:
			value = args[0]
			if not isinstance(value, (int, float)):
				continue
			ai.relation_system.set_relation(dominance=value)	
		elif command == "reset_mood":
			ai.emotion_system.reset_mood()	
		elif command == "consolidate_memories":
			ai.memory_system.consolidate_memories()
		elif command == "memories":
			print("Current memories:")
			for memory in ai.memory_system.get_short_term_memories():
				print(memory.format_memory())
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
		elif command == "wipe" or command == "reset":
			if os.path.exists(SAVE_PATH):
				choice = input("Really erase saved data and memories for this AI? Type 'yes' to erase data, or anything else to cancel: ")
				if choice.strip().lower() == "yes":
					os.remove(SAVE_PATH)
					input("The AI has been reset. Press enter to continue.")
					clear_screen()
					ai = AISystem()
					ai.on_startup()
		continue
	
	print()
	try:
		message = ai.send_message(msg)
	except Exception as e:
		import traceback
		traceback.print_exception(type(e), e, e.__traceback__)
		print("Oops! There was an error processing your input. Please try again in a moment.")
	else:
		ai.save(SAVE_PATH)
		print("AI: " + message)
	