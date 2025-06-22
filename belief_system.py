"""The system that manages the beliefs of the AI."""

from llm import MistralLLM

BELIEF_SYSTEM_PROMPT = """Generate a belief that would arise, given the memory.
The belief should be a sentence written from {name}'s perspective.

## Examples:

### Example 1
Memory:
```
You and your friends win a hockey championship after scoring the winning goal!
```
Belief: I'm a winner.

### Example 2
Memory:
```
You have been listening to music from a band called "Get Up and Glow"!
You continue to listen because of how much you love it.
```
Belief: Get Up and Glow is the best band ever!

### Example 3
Memory:
```
You have a LOT of homework to do, and you are so stressed out by the workload that you start to dislike homework.
```
Belief: Homework should be illegal.

### Other Examples

Other belief sentences can include (but are not limited to):
- I'm strong.
- I'm brave.
- I'm a really good friend.

## Input

Remember to generate a belief in first-person, from the AI's perspective.

Memory:
```
{memory}
```
Belief: """

class BeliefSystem:
	"""The system that manages the AI's beliefs"""
	model = MistralLLM("mistral-medium-latest")
	max_beliefs = 12

	def __init__(self, config):
		self.config = config
		self.beliefs = []

	def get_beliefs(self):
		"""Returns a list of the AI's current beliefs"""
		return [belief["content"] for belief in self.beliefs]

	def _generate_belief(self, memory, importance):
		name = self.config.name
		schema = {
			"type": "object",
			"properties": {
				"content": {"type":"string"},
				"importance": {"type": "number"}
			},
			"required": ["content", "importance"],
			"additionalProperties": False
		}
		prompt = BELIEF_SYSTEM_PROMPT.format(memory=memory, name=name)
		messages = [
			{
				"role": "system",
				"content": "You are a belief generator that generates a natural belief sentence " \
					f"in first-person POV given a memory, from {name}'s perspective. Generate a belief sentence, " \
					"and assign an importance score from 0.0 (trivial) to 1.0 (very important)."
			},
			{"role": "user", "content":prompt}
		]
		belief = self.model.generate(
			messages,
			temperature=1.0,
			schema=schema,
			return_json=True
		)
		belief["importance"] = (belief["importance"] + importance) / 2
		return belief

	def _has_belief(self, belief):
		return any(b["content"] == belief["content"] for b in self.beliefs)

	def _add_belief(self, belief):
		if len(self.beliefs) >= self.max_beliefs:
			min_importance = min(b["importance"] for b in self.beliefs)
			if belief["importance"] < min_importance:
				return False
			
		self.beliefs.append(belief)
		self.beliefs.sort(key=lambda b: b["importance"], reverse=True)
		if len(self.beliefs) > self.max_beliefs:  # Keep most important beliefs
			self.beliefs = self.beliefs[:self.max_beliefs]
		return True

	def generate_new_belief(self, memory, importance):
		"""Generates a new belief given a memory."""
		try:
			belief = self._generate_belief(memory, importance)
		except Exception as e:
			return
		if not self._has_belief(belief):
			self._add_belief(belief)

	def _tick(self, dt):
		for belief in self.beliefs:
			# Important memories decay slower
			half_life = 20 * 86400 * (1 + belief["importance"]**2 * 3)
			decay_factor = 0.5 ** (dt / half_life)
			belief["importance"] *= decay_factor

	def tick(self, delta):
		"""Ticks the belief system"""
		subtick = max(1, min(delta / 10, 10000))
		while delta > 0:
			subt = min(delta, subtick)
			self._tick(subt)
			delta -= subt
		
