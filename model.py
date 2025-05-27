import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import random
from typing import List, Dict, Optional


class ChatModel:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = "microsoft/DialoGPT-medium"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

        # dictionary to save dialogs
        self.chat_history = {}

        self.personalities_file = "personalities.json"
        self.personalities = self.load_personalities()
        self.default_personality = "friendly"

    def load_personalities(self) -> Dict[str, Dict[str, str]]:
        """Load AI personalities from file"""
        if os.path.exists(self.personalities_file):
            try:
                with open(self.personalities_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self.get_default_personalities()
        else:
            return self.get_default_personalities()

    def get_default_personalities(self) -> Dict[str, Dict[str, str]]:
        """Get default AI personalities"""
        return {
            "friendly": {
                "greeting": "Hello! How can I help you today?",
                "farewell": "Goodbye! Have a great day!",
                "thinking": "Let me think about that...",
                "error": "I'm sorry, I didn't understand that.",
                "positive": ["That's great!", "Wonderful!", "Excellent!", "Amazing!"],
                "negative": ["I'm sorry to hear that.", "That's unfortunate.", "I understand."],
                "neutral": ["I see.", "Interesting.", "Tell me more.", "Go on."]
            },
            "professional": {
                "greeting": "Good day. How may I assist you?",
                "farewell": "Thank you for your time. Have a productive day.",
                "thinking": "Processing your request...",
                "error": "I apologize, but I need more information.",
                "positive": ["Excellent work.", "Well done.", "Impressive.", "Noted."],
                "negative": ["I understand your concern.", "Let's address this issue.", "I see the problem."],
                "neutral": ["Understood.", "Proceed.", "Continue.", "Elaborate."]
            },
            "casual": {
                "greeting": "Hey there! What's up?",
                "farewell": "See you later! Take care!",
                "thinking": "Hmm, let me think...",
                "error": "Oops, I didn't catch that. Can you repeat?",
                "positive": ["Cool!", "Awesome!", "That's rad!", "Sweet!"],
                "negative": ["Bummer!", "That sucks!", "Sorry to hear that.", "That's rough."],
                "neutral": ["Yeah?", "And?", "Tell me more!", "What else?"]
            }
        }

    def save_personalities(self):
        """Save AI personalities to file"""
        with open(self.personalities_file, 'w', encoding='utf-8') as f:
            json.dump(self.personalities, f, ensure_ascii=False, indent=2)

    def add_personality(self, name: str, personality: Dict[str, str]):
        """Add new AI personality"""
        self.personalities[name] = personality
        self.save_personalities()

    def remove_personality(self, name: str):
        """Remove AI personality"""
        if name in self.personalities and name != self.default_personality:
            del self.personalities[name]
            self.save_personalities()

    def get_personality(self, name: str) -> Dict[str, str]:
        """Get AI personality by name"""
        return self.personalities.get(name, self.personalities[self.default_personality])

    def generate_response(self, user_id, message):
        # get user history
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []

        
        self.chat_history[user_id].append(message)

  
        context = " ".join(self.chat_history[user_id][-5:])  # last 5 messages

        #token
        inputs = self.tokenizer.encode(context + self.tokenizer.eos_token,
                                       return_tensors='pt').to(self.device)

        # genearte an answer
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8
            )

        
        response = self.tokenizer.decode(outputs[:, inputs.shape[-1]:][0],
                                         skip_special_tokens=True)

        
        self.chat_history[user_id].append(response)

        
        if len(self.chat_history[user_id]) > 10:
            self.chat_history[user_id] = self.chat_history[user_id][-10:]

        return response

    def clear_history(self, user_id):
        """Очищает историю диалога для конкретного пользователя"""
        if user_id in self.chat_history:
            self.chat_history[user_id] = []

    async def generate_response_with_personality(
            self,
            user_id: int,
            message: str,
            history: List[Dict[str, str]],
            personality: str = "friendly"
    ) -> str:
        """Generate AI response based on message and history"""
        # Get personality
        personality_data = self.get_personality(personality)

        # Simple response generation based on message content
        message = message.lower()

        # Check for greetings
        if any(word in message for word in ["hello", "hi", "hey", "greetings"]):
            return personality_data["greeting"]

        # Check for farewells
        if any(word in message for word in ["bye", "goodbye", "see you", "farewell"]):
            return personality_data["farewell"]

        # Check for questions
        if "?" in message:
            return random.choice(personality_data["neutral"])

        # Check for positive words
        if any(word in message for word in ["good", "great", "awesome", "excellent", "wonderful"]):
            return random.choice(personality_data["positive"])

        # Check for negative words
        if any(word in message for word in ["bad", "terrible", "awful", "horrible", "sad"]):
            return random.choice(personality_data["negative"])

        # Default response
        return random.choice(personality_data["neutral"])
