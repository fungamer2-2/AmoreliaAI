#pylint:disable=C0115
#pylint:disable=C0114
import json
import time
from datetime import datetime

from llm import MistralLLM
from const import *
from utils import (
	format_memories_to_string,
	time_since_last_message_string
)
from emotion_system import Emotion
from colored import Fore, Style


class ThoughtSystem:

	def __init__(
		self,
		config,
		emotion_system,
		memory_system,
		relation_system,
		personality_system
	):
		self.model = MistralLLM()
		self.config = config
		self.emotion_system = emotion_system
		self.memory_system = memory_system
		self.relation_system = relation_system
		self.personality_system = personality_system
		self.show_thoughts = True
		self.reflection_counter = 0
		self.last_reflection = datetime.now()
	
	def can_reflect(self):
		"""Determines whether the AI should reflect on its memories and gain insights."""
		return (
			self.memory_system.importance_counter >= 10
			and (datetime.now() - self.last_reflection).total_seconds() > 6 * 3600
			and len(self.memory_system.get_short_term_memories()) >= 5
		)
		
	def reflect(self):
		"""Performs 'reflection' - the AI can reflect on its memories to gain higher-level insights."""	
		recent_memories = self.memory_system.get_short_term_memories()
		memories_str = "\n".join(mem.format_memory() for mem in recent_memories)

		prompt = REFLECT_GEN_TOPICS.format(
			memories=memories_str
		)
		messages = [
			{"role":"system", "content":self.config.system_prompt},
			{"role":"user", "content":prompt}
		]
		print("Reflecting on memories...")
		questions = self.model.generate(
			messages,
			temperature=0.1,
			return_json=True
		)["questions"]

		for question in questions:
			print(f"Reflecting on '{question}'")
			relevant_memories = (
				self.memory_system.get_short_term_memories()
				+ self.memory_system.retrieve_long_term(question, 12)
			)
			memories_str = "\n".join(mem.format_memory() for mem in relevant_memories)
			insight_prompt = REFLECT_GEN_INSIGHTS.format(
				memories=memories_str,
				question=question
			)
			messages = [
				{"role":"system", "content":self.config.system_prompt},
				{"role":"user", "content":insight_prompt}
			]
			insights = self.model.generate(
				messages,
				temperature=0.1,
				return_json=True
			)["insights"]
			print("Insights gained:")
			for insight in insights:
				self.memory_system.remember(f"I gained an insight after reflection: {insight}", is_insight=True)
				print("- " + insight)
		self.memory_system.reset_importance()
		self.last_reflection = datetime.now()

	def _check_and_fix_thought_output(self, data):
		data = data.copy()
	
		data.setdefault("possible_user_emotions", [])
	
		data.setdefault("emotion_intensity", 5)
		data["emotion_intensity"] = max(1, min(10, int(data["emotion_intensity"])))
	
		data.setdefault("thoughts", [])
		data.setdefault("emotion", "Neutral")
		data.setdefault("emotion_reason", "I feel this way based on how the conversation has been going.")
		if data["emotion"] not in EMOTION_MAP:
			for emotion in EMOTION_MAP:
				if emotion.lower() == data["emotion"].lower():
					data["emotion"] = emotion
					break
			else:
				data["emotion"] = "Neutral"

		data.setdefault("next_action", "final_answer")
		data.setdefault("relationship_change", {"friendliness": 0.0, "dominance": 0.0})
		return data

	def think(self, messages, memories, recalled_memories, last_message):
		"""Generates the AI's internal thoughts and emotions"""
		memories_str = format_memories_to_string(
			memories,
			"You don't have any memories of this user yet!"
		)
		
		memory_emotion = Emotion()
		if recalled_memories:
			# Add emotion influence from recalled memories
			total_weight = 0.0
			for memory in recalled_memories:
				weight = memory.get_recency_factor(True)
				memory_emotion += memory.emotion * weight
				total_weight += weight
			memory_emotion /= total_weight
			self.emotion_system.add_emotion(memory_emotion * 0.3)

		content = messages[-1]["content"]

		img_data = None
		if isinstance(content, list):
			assert len(content) == 2
			assert content[0]["type"] == "text"
			assert content[1]["type"] == "image_url"
			text_content = content[0]["text"] + "\n\n((The user attached an image to this message - please see the attached image.))"
			img_data = content[1]
		else:
			text_content = content

		beliefs = self.memory_system.get_beliefs()
		if beliefs:
			belief_str = "\n".join(f"- {belief}" for belief in beliefs)
		else:
			belief_str = "None"
		
		appraisal = self.emotion_system.appraisal(messages, memories, beliefs)
		appraisal_str = ", ".join(
			f"{emotion} (Intensity {round(intensity*100)}%)"
			for emotion, intensity in appraisal
			if intensity >= 0.1	
		)
		appraisal_hint = ""
		if appraisal and appraisal_str:
			appraisal_hint = f"[This event makes {self.config.name} feel: {appraisal_str}]"
		
		last_interaction = time_since_last_message_string(last_message)
		prompt = THOUGHT_PROMPT.format(
			name=self.config.name,
			user_input=text_content,
			personality_summary=self.personality_system.get_summary(),
			mood_long_desc=self.emotion_system.get_mood_long_description(),
			curr_date=datetime.now().strftime("%a, %-m/%-d/%Y"),
			curr_time=datetime.now().strftime("%-I:%M %p"),
			mood_prompt=self.emotion_system.get_mood_prompt(),
			memories=memories_str,
			relationship_str=self.relation_system.get_string(),
			beliefs=belief_str,
			last_interaction=last_interaction,
			appraisal_hint=appraisal_hint
		)
		prompt_content = prompt
		if img_data:
			prompt_content = [
				{"type":"text", "text":prompt_content},
				img_data
			]

		thought_history = [
			{"role":"system", "content":self.config.system_prompt},
			{"role":"user", "content":"[START OF PREVIOUS CHAT HISTORY]"},
			*messages[:-1],
			{"role":"user", "content":"[END OF PREVIOUS CHAT HISTORY]"},
			{"role":"user", "content":prompt_content}
		]
		
		data = {}
		for _ in range(5):
			data = self.model.generate(
				thought_history,
				temperature=1.0,
				return_json=True,
				schema=THOUGHT_SCHEMA
			)
			
			if data.get("thoughts", []):
				break

		data = self._check_and_fix_thought_output(data)
		thought_history.append({
			"role": "assistant",
			"content": json.dumps(data, indent=4)
		})

		if self.show_thoughts:
			print("Thinking:")
			for thought in data["thoughts"]:
				print(Fore.magenta + thought['content'] + Style.reset)

		thoughts_query = " ".join(thought["content"] for thought in data["thoughts"])
	
		num_steps = 0
		
		# Let it continue thinking if necessary
		while data["next_action"].lower() == "continue_thinking":
			num_steps += 1
			added_context = ""
			relevant_memories = self.memory_system.long_term.retrieve(thoughts_query, MEMORY_RETRIEVAL_TOP_K)
			if relevant_memories:
				added_context = ADDED_CONTEXT_TEMPLATE.format(
					"\n".join(mem.format_memory() for mem in memories)
				)

			thought_history.append({
				"role": "user",
				"content": HIGHER_ORDER_THOUGHTS.format(added_context=added_context)
			})
			new_data = self.model.generate(
				thought_history,
				temperature=1.0,
				return_json=True,
				schema=THOUGHT_SCHEMA
			)
			new_data = self._check_and_fix_thought_output(new_data)
			thought_history.append({
				"role": "assistant",
				"content": json.dumps(new_data, indent=4)
			})
			thoughts_query = " ".join(thought["content"] for thought in new_data["thoughts"])
	
			if self.show_thoughts:
				print()
				for thought in new_data["thoughts"]:
					print(Fore.magenta + thought['content'] + Style.reset)
		
			all_thoughts = data["thoughts"] + new_data["thoughts"]
			data = new_data.copy()
			data["thoughts"] = all_thoughts
			if num_steps >= MAX_THOUGHT_STEPS:
				break

		if not appraisal and data["emotion"] != "Neutral":
			appraisal = [(data["emotion"], data["emotion_intensity"])]
		
		total_emotion = Emotion()	
		for emotion, intensity in appraisal:
			total_emotion += self.emotion_system.experience_emotion(emotion, intensity)
		
		relation_change = data["relationship_change"]
		data["emotion"] = appraisal_hint
		data["emotion_obj"] = total_emotion
		self.relation_system.change_relationship(
			relation_change.get("friendliness", 0.0),
			relation_change.get("dominance", 0.0)
		)
		return data
