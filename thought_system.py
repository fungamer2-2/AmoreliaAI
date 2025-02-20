from datetime import datetime

from llm import MistralLLM
from const import *
from utils import get_model_to_use

import json


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
		return (
			self.memory_system.importance_counter >= 9
			and (datetime.now() - self.last_reflection).total_seconds() > 3 * 3600
			and len(self.memory_system.get_short_term_memories()) >= 10
		)
			
	def reflect(self):	
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
				self.memory_system.get_memories()
				+ self.memory_system.long_term.retrieve(question, k=12)
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
				self.memory_system.remember(f"I gained an insight after reflection: {insight}")
				print("- " + insight)
		self.memory_system.reset_importance()
		self.last_reflection = datetime.now()

	def _check_and_fix_thought_output(self, data):
		data = data.copy()
		data.setdefault("emotion_intensity", 5)
		data["emotion_intensity"] = int(data["emotion_intensity"])
		
		data.setdefault("thoughts", [])
		data.setdefault("emotion", "Neutral")
		data.setdefault("emotion_reason", "I feel this way based on how the conversation has been going.")
		
		if data["emotion"] not in EMOTION_MAP:
			for em in EMOTION_MAP:
				if em.lower() == data["emotion"].lower():
					data["emotion"] = em
					break
			else:
				data["emotion"] = "Neutral"
		
		data.setdefault("next_action", "final_answer")
		
		return data		
		
	def think(self, messages, memories):		
		memories_str = (
			"\n".join(mem.format_memory() for mem in memories)
			if memories 
			else "You don't have any memories of this user yet!"
		)
		
		content = messages[-1]["content"]
		
		img_data = None
		if isinstance(content, list):
			assert len(content) == 2
			assert content[0]["type"] == "image_url"
			assert content[1]["type"] == "text"
			text_content = content[1]["text"] + "\n\n((The user attached an image to this message))"
			img_data = content[0]
		else:
			text_content = content
		
		prompt = THOUGHT_PROMPT.format(
			name=self.config.name,
			user_input=text_content,
			personality_summary=self.personality_system.get_summary(),
			mood_long_desc=self.emotion_system.get_mood_long_description(),
			curr_date=datetime.now().strftime("%a, %-m/%-d/%Y"),
			curr_time=datetime.now().strftime("%-I:%M %p"),
			mood_prompt=self.emotion_system.get_mood_prompt(),
			memories=memories_str,
			relationship_str=self.relation_system.get_string()
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
		model = get_model_to_use(messages)
			
		data = model.generate(
			thought_history,
			temperature=0.8,
			return_json=True,
			schema=THOUGHT_SCHEMA
		)
		
		data = self._check_and_fix_thought_output(data)
		thought_history.append({
			"role": "assistant",
			"content": json.dumps(data, indent=4)
		})
		
		if self.show_thoughts:
			print(f"{self.config.name}'s thoughts:")
			for thought in data["thoughts"]:
				print(f"- {thought}")
			print()
		
		num_steps = 0
		while data["next_action"].lower() == "continue":
			num_steps += 1
			thought_history.append({
				"role": "user",
				"content": HIGHER_ORDER_THOUGHTS
			})
			new_data = model.generate(
				thought_history,
				temperature=0.8,
				return_json=True,
				schema=THOUGHT_SCHEMA
			)
			new_data = self._check_and_fix_thought_output(new_data)	
			thought_history.append({
				"role": "assistant",
				"content": json.dumps(new_data, indent=4)
			})
			if self.show_thoughts:			
				for thought in new_data["thoughts"]:
					print(f"- {thought}")
				print()
			
			all_thoughts = data["thoughts"] + new_data["thoughts"]
			data = new_data.copy()
			data["thoughts"] = all_thoughts
			if num_steps >= MAX_THOUGHT_STEPS:
				break
		
		self.emotion_system.experience_emotion(
			data["emotion"],
			data["emotion_intensity"]/10
		)
		
		return data
