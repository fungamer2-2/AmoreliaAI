# HumanlikeAI
An AI companion designed for more humanlike interactions

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/fungamer2-2/HumanlikeAI)
![GitHub last commit](https://img.shields.io/github/last-commit/fungamer2-2/HumanlikeAI)
![GitHub License](https://img.shields.io/github/license/fungamer2-2/HumanlikeAI)


Currently uses [Mistral AI](https://mistral.ai) models. To use this project, you'll need a Mistral AI API key, and store it under `MISTRAL_API_KEY` in a `.env` file

## Thought system

Before responding, the AI is prompted to generate a list of thoughts from its perspective. 

## Emotion system

The emotion system is based on the PAD (Pleasure-Arousal-Dominance) state model. Interactions with the AI will elicit emotions, which affect its mood. Its current emotions may affect its responses.

## Memory system

The AI companion also has a long-term memory system to recall relevant memories from previous conversations. It includes two types of memory: short-term and long-term.
