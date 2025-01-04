from datetime import datetime

from llm import MistralLLM
from emotion_system import Emotion
from const import *
import json

class ThoughtSystem:
	
	def __init__(
		self,
		emotion_system,
		memory_system,
		relation_system
	):
		self.model = MistralLLM("mistral-large-latest")
		self.emotion_system = emotion_system
		self.memory_system = memory_system
		self.relation_system = relation_system
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
		
		data.setdefault("further_thought_needed", False)
		
		return data		
		
		
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
			long_term=long_term,
			relationship_str = self.relation_system.get_string()
		)
		 
		thought_history = [
			{"role":"system", "content":AI_SYSTEM_PROMPT},
			{"role":"user", "content":prompt}
		]
		data = self.model.generate(
			thought_history,
			temperature=0.7,
			presence_penalty=0.5,
			return_json=True
		)
		
		data = self._check_and_fix_thought_output(data)
		
		thought_history.append({
			"role": "assistant",
			"content": json.dumps(data, indent=4)
		})
		
		if self.show_thoughts:
			print("AI thoughts:")
			for thought in data["thoughts"]:
				print(f"- {thought}")
			print()
			
		continue_thinking = data["further_thought_needed"]
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
				print(new_data)
				print("Higher-order thoughts:")
				for thought in new_data["thoughts"]:
					print(f"- {thought}")
				print()
			
			all_thoughts = data["thoughts"] + new_data["thoughts"]
			data = new_data.copy()
			data["thoughts"] = all_thoughts
			continue_thinking = data["further_thought_needed"] and num_steps < max_steps
			
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
