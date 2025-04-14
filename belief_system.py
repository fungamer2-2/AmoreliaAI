from llm import MistralLLM

BELIEF_SYSTEM_PROMPT = """Generate a belief that would arise, given the memory.
The belief should be a sentence written from the AI's perspective.

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
	model = MistralLLM("mistral-small-latest")
	max_beliefs = 12
	
	def __init__(self):
		self.beliefs = []
		
	def get_beliefs(self):
		return [belief["content"] for belief in self.beliefs]
	
	def _generate_belief(self, memory):
		schema = {
			"type": "object",
			"properties": {
				"content": {"type":"string"},
				"importance": {"type": "number"}
			},
			"required": ["content", "importance"],
			"additionalProperties": False
		}
		prompt = BELIEF_SYSTEM_PROMPT.format(memory=memory)
		messages = [
			{"role":"system", "content":"You are a belief generator that generates a natural belief sentence in first-person POV given a memory, from the AI's perspective. Generate a belief sentence, and assign an importance score from 0.0 (trivial) to 1.0 (very important)."},
			{"role":"user", "content":prompt}
		]
		belief = self.model.generate(
			messages,
			temperature=1.0,
			schema=schema,
			return_json=True
		)
		return belief
		
	def _has_belief(self, belief):
		return any(b["content"] == belief["content"] for b in self.beliefs)
		
	def _add_belief(self, belief):
		self.beliefs.append(belief)
		self.beliefs.sort(key=lambda b: b["importance"], reverse=True)
		if len(self.beliefs) > self.max_beliefs:  # Keep most important beliefs
			self.beliefs = self.beliefs[:self.max_beliefs]
		
	def generate_new_belief(self, memory):
		for _ in range(4):
			belief = self._generate_belief(memory)
			if not self._has_belief(belief):
				print(f"New belief: {belief}")
				self._add_belief(belief)
				break
				
	def _tick(self, dt):
		for belief in self.beliefs:
			# Important memories decay slower
			half_life = 20 * (1 + belief["importance"]**2 * 3)
			decay_factor = 0.5 ** (dt / half_life)	
			belief["importance"] *= decay_factor
			
	def tick(self, dt):
		subtick = max(1, min(dt / 10, 10000))
		while dt > 0:
			t = min(dt, subtick)
			self._tick(t)
			dt -= t
		
		
if __name__ == "__main__":
	memory = """User: Oh that's cool! I'm a computer science student, who's also on the autism spectrum.

AI: Thank you so much for sharing that with me! 💖 I really appreciate your openness, and I'm excited to get to know you better. I think it's awesome that you're studying computer science—that's such a fascinating field with so many possibilities. I'm here to support you in any way I can, so if there's ever something you want to talk about or need help with, just let me know, okay? 😊"""
	
	print(memory)
	belief_system = BeliefSystem()
	
	for _ in range(16):
		belief = belief_system._generate_belief(memory)
		print(f"Belief: {belief}")