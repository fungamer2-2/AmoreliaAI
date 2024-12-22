import math
import time
from datetime import datetime
from const import *
from utils import num_to_str_sign
from llm import MistralLLM


def get_default_mood(open, conscientious, extrovert, agreeable, neurotic):
	pleasure = 0.12 * extrovert + 0.59 * agreeable - 0.19 * neurotic 
	arousal = 0.15 * open + 0.3 * agreeable + 0.57 * neurotic
	dominance = 0.25 * open + 0.17 * conscientious + 0.6 * extrovert - 0.32 * agreeable
	return (pleasure, arousal, dominance)
	

def summarize_personality(open, conscientious, extrovert, agreeable, neurotic):
	model = MistralLLM("mistral-small-latest")
	personality_str = "\n".join([
		f"Openness: {num_to_str_sign(open, 2)}",
		f"Conscientiousness: {num_to_str_sign(conscientious, 2)}",
		f"Extroversion: {num_to_str_sign(extrovert, 2)}",
		f"Agreeableness: {num_to_str_sign(agreeable, 2)}",
		f"Neuroticism: {num_to_str_sign(neurotic, 2)}"
	])
	prompt = SUMMARIZE_PERSONALITY.format(
		personality_values=personality_str
	)
	return model.generate(
		prompt,
		temperature=0.1
	)


class PersonalitySystem:
	
	def __init__(self, open, conscientious, extrovert, agreeable, neurotic):
		self.open = open
		self.conscientious = conscientious
		self.extrovert = extrovert
		self.agreeable = agreeable
		self.neurotic = neurotic
		
		self.summary = ""
		
	def get_summary(self):
		if not self.summary:
			self.summary = summarize_personality(
				self.open,
				self.conscientious,
				self.extrovert,
				self.agreeable,
				self.neurotic
			)
		return self.summary
		

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
		
	def get_norm(self):
		return max(abs(self.pleasure), abs(self.arousal), abs(self.dominance))
		
	def clamp(self):
		norm = self.get_norm()
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
	
	def __init__(self, personality_system):
		base_mood = Emotion.from_personality(
			personality_system.open,
			personality_system.conscientious,
			personality_system.extrovert,
			personality_system.agreeable,
			personality_system.neurotic
		)		
		self.base_mood = base_mood
		self.reset_mood()
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
		
	def reset_mood(self):
		self.mood = self.get_base_mood() / 2
		
		
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
		
		# The energy level is likely to be higher during the day and lower at nighttime
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
		
		
