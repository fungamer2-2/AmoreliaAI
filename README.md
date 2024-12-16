# HumanlikeAI

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/fungamer2-2/HumanlikeAI)
![GitHub last commit](https://img.shields.io/github/last-commit/fungamer2-2/HumanlikeAI)
![GitHub License](https://img.shields.io/github/license/fungamer2-2/HumanlikeAI)

An AI companion designed for more humanlike interactions. The goal is to create a humanlike AI system that isn't just an assistant, but a virtual companion that you can form connections with.

Currently uses [Mistral AI](https://mistral.ai) models. To use this project, you'll need a Mistral AI API key, and store it under `MISTRAL_API_KEY` in a `.env` file.

1. Download this project
2. Make sure that [Python](https://python.org) is installed
3. Obtain a Mistral API key at <https://console.mistral.ai/>
4. Store the API key in a `.env` file in the project directory
5. Run `main.py`


## Thought system

Before responding, the AI is prompted to generate a list of thoughts from its perspective. This helps make it a bit more realistic, and think as if it truly had its own personality.

## Emotion system

The emotion system is based on the PAD (Pleasure-Arousal-Dominance) state model. Interactions with the AI will elicit emotions, which affect its mood. Its current emotions may affect its responses.

## Memory system

The AI companion also has a long-term memory system to recall relevant memories from previous conversations. It includes two types of memory: short-term and long-term.

- Short-term memory: Memories that the AI has either experienced recently or recalled recently. This has a limited capacity, and any memories that get flushed out of short-term memory are sent to long-term memory.
- Long term memory: Stores the memories and experiences to be retrieved whenever they become relevant. Recalled memories return to short-term memory.
