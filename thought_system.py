from datetime import datetime

from llm import MistralLLM
from emotion_system import Emotion
from const import *
import json

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
		thought_history.append({
			"role": "assistant",
			"content": json.dumps(data, indent=4)
		})
		intensity = int(data.get("emotion_intensity", 5))
		emotion = data.get("emotion", "")
		insights = data.get("high_level_insights", [])
		
		if emotion not in EMOTION_MAP:
			for em in EMOTION_MAP:
				if em.lower() == emotion.lower():
					data["emotion"] = em
					break
			else:
				data["emotion"] = "Neutral"
				
		data["emotion_intensity"] = intensity
		data.setdefault("emotion_reason", "I feel this way based on how the conversation has been going.")
		
		
		self.emotion_system.experience_emotion(
			data["emotion"],
			intensity/10
		)
		
		if self.show_thoughts:
			print("AI thoughts:")
			for thought in data["thoughts"]:
				print(f"- {thought}")
			print()
			
			#print(f"Emotion: {data['emotion']}")
#		
#			print(f"Emotion influence: {data['emotion_influence']}")
		
		if insights:
			# Add new insights gained into memory
			print("Insights gained:")
			for insight in insights:
				print(f"- {insight}")
				self.memory_system.remember(f"I gained an insight while chatting with the user: {insight}")
			print()
			
		order = int(data.get("further_thought_needed", 0))
		#print(f"Order: {order}")
		for _ in range(order):
			thought_history.append({
				"role": "user",
				"content": HIGHER_ORDER_THOUGHTS
			})
			new_thoughts = self.model.generate(
				thought_history,
				temperature=0.7,
				presence_penalty=0.5,
				return_json=True
			)
			thought_history.append({
				"role": "assistant",
				"content": json.dumps(data, indent=4)
			})
			if self.show_thoughts:
				print("Higher-order thoughts:")
				for thought in new_thoughts["thoughts"]:
					print(f"- {thought}")
				print()
			data["thoughts"].extend(new_thoughts["thoughts"])
			
		return data
