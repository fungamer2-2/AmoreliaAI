from datetime import datetime

from llm import MistralLLM
from emotion_system import Emotion
from const import *
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
		self.model = MistralLLM("mistral-large-latest")
		self.config = config
		self.emotion_system = emotion_system
		self.memory_system = memory_system
		self.relation_system = relation_system
		self.personality_system = personality_system
		self.show_thoughts = True
		
	def _check_and_fix_thought_output(self, data):
		data = data.copy()
		data.setdefault("emotion_intensity", 5)
		data["emotion_intensity"] = int(data["emotion_intensity"])
		
		data.setdefault("emotion", "Neutral")
		data.setdefault("high_level_insights", [])
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
		role_map = {
			"user": "User",
			"assistant": self.config.name
		}
		history_str = "\n\n".join(
			f"{role_map[msg['role']]}: {msg['content']}"
			for msg in messages[:-1]
		)
		mood_prompt = self.emotion_system.get_mood_prompt()
		mood = self.emotion_system.mood
		
		memories_str = "\n".join(mem.format_memory() for mem in memories)
		
		prompt = THOUGHT_PROMPT.format(
			history_str=history_str,
			name=self.config.name,
			user_input=messages[-1]["content"],
			personality_summary=self.personality_system.get_summary(),
			mood_long_desc=self.emotion_system.get_mood_long_description(),
			curr_date=datetime.now().strftime("%a, %-m/%-d/%Y"),
			curr_time=datetime.now().strftime("%-I:%M %p"),
			mood_prompt=mood_prompt,
			memories=memories_str,
			relationship_str = self.relation_system.get_string()
		)
		 
		thought_history = [
			{"role":"system", "content":self.config.system_prompt},
			{"role":"user", "content":prompt}
		]
		data = self.model.generate(
			thought_history,
			temperature=0.8,
			return_json=True
		)
		
		data = self._check_and_fix_thought_output(data)
		#print(data["possible_user_emotions"])
		thought_history.append({
			"role": "assistant",
			"content": json.dumps(data, indent=4)
		})
		
		if self.show_thoughts:
			print(f"{self.config.name}'s thoughts:")
			for thought in data["thoughts"]:
				print(f"- {thought}")
			print()
			
		continue_thinking = data["next_action"].lower() == "continue"
		max_steps = 5
		
		num_steps = 0
		while continue_thinking:
			num_steps += 1
			thought_history.append({
				"role": "user",
				"content": HIGHER_ORDER_THOUGHTS
			})
			new_data = self.model.generate(
				thought_history,
				temperature=0.7,
				presence_penalty=0.5,
				return_json=True
			)
			new_data = self._check_and_fix_thought_output(new_data)
			thought_history.append({
				"role": "assistant",
				"content": json.dumps(new_data, indent=4)
			})
			if self.show_thoughts:
				print("Higher-order thoughts:")
				for thought in new_data["thoughts"]:
					print(f"- {thought}")
				print()
			
			all_thoughts = data["thoughts"] + new_data["thoughts"]
			data = new_data.copy()
			data["thoughts"] = all_thoughts
			continue_thinking = data["next_action"].lower() == "continue" and num_steps < max_steps
			
		intensity = data["emotion_intensity"]
		emotion = data["emotion"]
		insights = data["high_level_insights"]
		
		self.emotion_system.experience_emotion(
			data["emotion"],
			intensity/10
		)
		
		if insights:
			# Add new insights gained into memory
			print("Insights gained:")
			for insight in insights:
				print(f"- {insight}")
				self.memory_system.remember(f"I gained an insight while chatting with the user: {insight}")
			print()
		
		return data
