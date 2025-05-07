"""The main module that runs the AI."""

import copy
import os
import traceback
import json
import pickle
from collections import deque
from datetime import datetime

from colored import Fore, Style
from pydantic import BaseModel, Field
from llm import MistralLLM
from utils import (
	clear_screen,
	get_model_to_use,
	is_image_url,
	format_memories_to_string,
	time_since_last_message_string
)
from emotion_system import (
	EmotionSystem,
	PersonalitySystem,
	RelationshipSystem
)
from memory_system import MemorySystem
from thought_system import ThoughtSystem

from const import (
	AI_SYSTEM_PROMPT,
	USER_TEMPLATE,
	SAVE_PATH
)

class MessageBuffer:
	"""A buffer that stores the most recent messages in the conversation, 
	flushing out older messages if they exceed the limit."""

	def __init__(self, max_messages):
		self.max_messages = max_messages
		self.messages = deque(maxlen=max_messages)
		self.system_prompt = ""

	def set_system_prompt(self, prompt):
		"""Sets the system prompt."""
		self.system_prompt = prompt.strip()

	def add_message(self, role, content):
		"""Adds a message to the buffer."""
		self.messages.append({"role": role, "content": content})
	
	def pop(self):
		"""Removes and returns the last message."""
		return self.messages.pop()

	def flush(self):
		"""Clears the buffer, removing all messages."""
		self.messages.clear()
	
	def to_list(self, include_system_prompt=True):
		"""Converts the buffer to a list of messages.
		The system prompt is included by default, but you can set include_system_prompt=False
		to not include it."""
		history = []
		if include_system_prompt and self.system_prompt:
			history.append({"role":"system", "content":self.system_prompt})
		history.extend(msg.copy() for msg in self.messages)
		return history


GENERATE_USER_RESPONSES_PROMPT = """# Task

Given the following conversation, please suggest 3 to 5 possible responses that the HUMAN could respond to the last AI message given the conversation context.
Try to match the human's tone and style as closely as possible. 

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

Today is {date}. The current time is {time}

Here is the conversation history so far:

```
{conversation_history}
```

Remember, try to match the human's tone and style as closely as possible. 

Possible **HUMAN** responses:"""
	
from datetime import datetime
def suggest_responses(conversation):
	"""Generates a list of potential user responses given the conversation history."""
	role_map = {
		"user": "HUMAN",
		"assistant": "AI"
	}
	history_str = "\n\n".join(
		f"{role_map[msg['role']]}: {msg['content']}"
		for msg in conversation
		if msg["role"] != "system"
	)
	now = datetime.now()
	model = MistralLLM("mistral-small-latest")
	prompt = GENERATE_USER_RESPONSES_PROMPT.format(
		conversation_history=history_str,
		date=now.strftime("%a, %-m/%-d/%Y"),
		time=now.strftime("%-I:%M %p")
	)

	data = model.generate(
		prompt,
		temperature=0.7,
		return_json=True
	)
	return data["possible_responses"]


class PersonalityConfig(BaseModel):
	"""The config for the personality of the AI"""
	open: float = Field(ge=-1.0, le=1.0)
	conscientious: float = Field(ge=-1.0, le=1.0)
	agreeable: float = Field(ge=-1.0, le=1.0)
	extrovert: float = Field(ge=-1.0, le=1.0)
	neurotic: float = Field(ge=-1.0, le=1.0)


class AIConfig(BaseModel):
	"""The config for the AI"""
	name: str = Field(default="Amorelia")
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
	"""The AI system, which contains various subsystems influencing the AI"""

	def __init__(self, config=None):
		config = config or AIConfig()
		personality = config.personality
	
		self.config = config
		self.personality_system = PersonalitySystem(
			openness=personality.open,
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

		self.num_messages = 0
		self.last_message = None
		self.last_recall_tick = datetime.now()
		self.last_tick = datetime.now()

		self.buffer = MessageBuffer(20)
		self.buffer.set_system_prompt(config.system_prompt)
		
	def set_config(self, config):
		"""Updates the config"""
		self.memory_system.config = config
		self.memory_system.belief_system.config = config
		self.thought_system.config = config
		personality = config.personality
		self.personality_system = PersonalitySystem(
			openness=personality.open,
			conscientious=personality.conscientious,
			extrovert=personality.extrovert,
			agreeable=personality.agreeable,
			neurotic=personality.neurotic
		)

	def get_message_history(self, include_system_prompt=True):
		"""Gets the current conversation history."""
		return self.buffer.to_list(include_system_prompt)

	def on_startup(self):
		"""Runs when the AI system is loaded."""
		self.buffer.flush()
		self.last_tick = datetime.now()
		self.tick()

	def _image_to_description(self, image_url):
		messages = [
			{"role":"system", "content":self.config.system_prompt},
			{
				"role":"user",
				"content": [
					{
						"type":"image_url",
						"image_url": image_url
					},
					{
						"type": "text",
						"text": "Please describe in detail what you see in this image. " \
							"Make sure to include specific details, such as style, colors, etc."
					}
				]
			}
		]
		model = MistralLLM("pixtral-large-latest")
		return model.generate(
			messages,
			temperature=0.1,
			max_tokens=1024
		)

	def _input_to_memory(self, user_input, ai_response, attached_image=None):
		user_msg = ""
		if attached_image:
			description = self._image_to_description(attached_image)
			user_msg += f'<attached_img url="{attached_image}">Description: {description}</attached_img>\n'

		user_msg += user_input

		return f"User: {user_msg}\n\n{self.config.name}: {ai_response}"
		
	def _get_format_data(self, content, thought_data, memories):
		now = datetime.now()
		user_emotions = thought_data["possible_user_emotions"]
		user_emotion_list_str =  ", ".join(user_emotions)
		if user_emotions:
			user_emotion_str = (
				"The user appears to be feeling the following emotions: "
				+ user_emotion_list_str
			)
		else:
			user_emotion_str = "The user doesn't appear to show any strong emotion."

		thought_str = "\n".join("- " + thought for thought in thought_data["thoughts"])
		beliefs = self.memory_system.get_beliefs()
		if beliefs:
			belief_str = "\n".join(f"- {belief}" for belief in beliefs)
		else:
			belief_str = "None"
		return {
			"name": self.config.name,
			"personality_summary": self.personality_system.get_summary(),
			"user_input": content,
			"ai_thoughts": thought_str,
			"emotion": thought_data["emotion"],
			"emotion_reason": thought_data["emotion_reason"],
			"emotion_influence": thought_data["emotion_influence"],
			"memories": format_memories_to_string(
				memories,
				"You don't have any memories of this user yet!"
			),
			"curr_date": now.strftime("%a, %-m/%-d/%Y"),
			"curr_time": now.strftime("%-I:%M %p"),
			"user_emotion_str": user_emotion_str,
			"beliefs": belief_str,
			"mood_prompt": self.emotion_system.get_mood_prompt(),
			"last_interaction": time_since_last_message_string(self.last_message)
		}

	def send_message(self, user_input: str, attached_image=None, return_json=False):
		"""Sends a message to the AI, and returns the response."""
		self.tick()
		
		self.last_recall_tick = datetime.now()
		self.buffer.set_system_prompt(self.config.system_prompt)

		content = user_input
		if attached_image is not None:
			content = [
				{
					"type": "image_url",
					"image_url": attached_image
				},
				{
					"type": "text",
					"text": user_input
				}
			]
		self.buffer.add_message("user", content)

		history = self.get_message_history()

		memories, recalled_memories = self.memory_system.recall_memories(history)
		memories.sort(key=lambda memory: memory.timestamp)

		thought_data = self.thought_system.think(
			self.get_message_history(False),
			memories,
			recalled_memories,
			self.last_message
		)

		content = history[-1]["content"]

		img_data = None
		if isinstance(content, list):
			assert len(content) == 2
			assert content[0]["type"] == "image_url"
			assert content[1]["type"] == "text"
			text_content = content[1]["text"] + "\n\n((The user attached an image to this message))"
			img_data = content[0]
		else:
			text_content = content

		prompt_content = USER_TEMPLATE.format(
			**self._get_format_data(text_content, thought_data, memories)
		)
		if img_data:
			prompt_content = [
				img_data,
				{"type":"text", "text":prompt_content}
			]

		history[-1]["content"] = prompt_content

		model = get_model_to_use(history)

		response = model.generate(
			history,
			temperature=0.8,
			return_json=return_json
		)

		self.memory_system.remember(
			self._input_to_memory(user_input, response, attached_image),
			emotion=thought_data["emotion_obj"]
		)
		self.last_message = datetime.now()
		self.tick()
		new_response = response
		if return_json:
			response = json.dumps(new_response, indent=2)
		self.buffer.add_message("assistant", new_response)
		return response

	def set_thought_visibility(self, shown: bool):
		"""Sets the flag for whether or not to show the AI's internal thoughts."""
		self.thought_system.show_thoughts = shown

	def get_mood(self):
		"""Gets the AI's current mood."""
		return self.emotion_system.mood
		
	def get_beliefs(self):
		"""Gets the AI's beliefs"""
		return self.memory_system.get_beliefs()

	def set_mood(self, pleasure=None, arousal=None, dominance=None):
		"""Sets the AI's current mood. All parameters are optional, but if none are specified, 
		resets the AI's mood to its baseline level."""
		if pleasure is None and arousal is None and dominance is None:
			self.emotion_system.reset_mood()
		else:
			self.emotion_system.set_emotion(
				pleasure=pleasure,
				arousal=arousal,
				dominance=dominance
			)
			
	def set_relation(self, friendliness=None, dominance=None):
		"""Sets the AI's relationship with the user"""
		self.relation_system.set_relation(
			friendliness=friendliness,
			dominance=dominance
		)

	def tick(self):
		"""Runs a tick to update the AI's systems"""
		now = datetime.now()
		delta = (now - self.last_tick).total_seconds()
		self.emotion_system.tick()
		if self.thought_system.can_reflect():
			self.thought_system.reflect()
		self.memory_system.tick(delta)
		
		if (now - self.last_recall_tick).total_seconds() > 2 * 3600:
			self.memory_system.surface_random_thoughts()
			print("Random thoughts surfaced")
			self.last_recall_tick = now
		self.last_tick = now
		
	def save(self, path):
		"""Saves the AI system to the path"""
		with open(path, "wb") as file:
			pickle.dump(self, file)
	
	@staticmethod
	def load(path):
		"""Loads the AI system from the path. Returns None if it doesn't exist."""
		if os.path.exists(path):
			with open(path, "rb") as file:
				return pickle.load(file)
		else:
			return None
			
	@classmethod
	def load_or_create(cls, path):
		"""Loads the AI system from the path, or creates it if it doesn't exist."""
		ai_system = cls.load(path)
		if ai_system is None:
			ai_system = AISystem()
			print("AI system initialized.")
		else:
			print("AI loaded.")
	
		ai_system.on_startup()
		return ai_system


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
	"""Parses a command into its arguments"""
	split = string.split(None, 1)
	if len(split) == 2:
		command, remaining = split
	else:
		command, remaining = string, ""
	args = remaining.split()
	return command, _parse_args(args)
	

# TODO: Add a user profile system


def main():
	"""The main method"""
	attached_image = None
	ai = AISystem.load_or_create(SAVE_PATH)
	print(f"{Fore.yellow}Note: It's recommended not to enter any sensitive information.{Style.reset}")
	
	while True:
		ai.tick()
		ai.emotion_system.print_mood()
		if attached_image:
			print(f"Attached image: {attached_image}")
		msg = input("User: ").strip()
		if not msg:
			ai.save(SAVE_PATH)	
			continue
			
		if msg.startswith("/"):
			command, args = command_parse(msg[1:])
			if command == "set_pleasure" and len(args) == 1:
				value = args[0]
				if not isinstance(value, (int, float)):
					continue
				ai.set_mood(pleasure=value)
			if command == "set_arousal" and len(args) == 1:
				value = args[0]
				if not isinstance(value, (int, float)):
					continue
				ai.set_mood(arousal=value)
			elif command == "set_dominance" and len(args) == 1:
				value = args[0]
				if not isinstance(value, (int, float)):
					continue
				ai.set_mood(dominance=value)
			elif command == "set_relation_friendliness" and len(args) == 1:
				value = args[0]
				if not isinstance(value, (int, float)):
					continue
				ai.set_relation(friendliness=value)
			elif command == "set_relation_dominance" and len(args) == 1:
				value = args[0]
				if not isinstance(value, (int, float)):
					continue
				ai.set_relation(dominance=value)
			elif command == "show_thoughts":
				ai.set_thought_visibility(True)
			elif command == "hide_thoughts":
				ai.set_thought_visibility(False)
			elif command == "reset_mood":
				ai.emotion_system.reset_mood()
			elif command == "consolidate_memories":
				ai.memory_system.consolidate_memories()
			elif command == "attach_image" and len(args) == 1:
				url = args[0]
				if not isinstance(url, str):
					continue
				if is_image_url(url):
					attached_image = url
				else:
					print("Error: Not a valid image url")
			elif command == "detach_image":
				attached_image = None
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
			elif command in ["wipe", "reset"]:
				if os.path.exists(SAVE_PATH):
					choice = input(
						"Really erase saved data and memories for this AI? "
						"Type 'yes' to erase data, or anything else to cancel: "
					)
					if choice.strip().lower() == "yes":
						os.remove(SAVE_PATH)
						input("The AI has been reset. Press enter to continue.")
						clear_screen()
						ai = AISystem()
						ai.on_startup()
			elif command == "beliefs":
				beliefs = ai.get_beliefs()
				if beliefs:
					print("The following beliefs have been formed:")
					for belief in beliefs:
						print("- " + belief)
				else:
					print("No beliefs have been formed yet")
			elif command == "configupdate":
				new_config = AIConfig()
				ai.set_config(new_config)
				ai.save(SAVE_PATH)
				print("Config updated and saved!")
			else:
				print(f"Invalid command '/{command}'")
			continue
	
		print()
		
		backup_ai = copy.deepcopy(ai)
		try:
			message = ai.send_message(msg, attached_image=attached_image)
		except Exception as e:  # pylint: disable=W0718,C0103
			ai = backup_ai  # Restore in case something changed before the error
			traceback.print_exception(type(e), e, e.__traceback__)
			print("An error occurred. Please try again in a moment.")
			print("If the issue persists, please open an issue on GitHub: https://github.com/fungamer2-2/HumanlikeAI/issues/new?template=bug_report.md")
		else:
			print(f"{ai.config.name}: " + message)
			ai.save(SAVE_PATH)
			attached_image = None
		

if __name__ == "__main__":
	main()
