import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
print("Downloading required NLTK data...")
try:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")
    raise

class SimpleChatbot:
    def __init__(self, intents_file='intents_final.json'):
        """
        Initialize the chatbot with the given intents file.
        The intents file should be a JSON file with 'intents' as the root key.
        
        Args:
            intents_file (str): Path to the JSON file containing the intents
        """
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(tokenizer=self.lemmatize_sentence, stop_words='english')
        self.intents = self.load_intents(intents_file)
        self.prepare_data()
    
    def load_intents(self, intents_file):
        """Load intents from a JSON file."""
        with open(intents_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data['intents']
    
    def lemmatize_sentence(self, sentence):
        """Tokenize and lemmatize a sentence."""
        words = nltk.word_tokenize(sentence.lower())
        return [self.lemmatizer.lemmatize(word) for word in words]
    
    def prepare_data(self):
        """Prepare the training data from intents."""
        self.tags = []
        self.patterns = []
        self.responses = {}
        
        for intent in self.intents:
            tag = intent['tag']
            self.tags.append(tag)
            self.responses[tag] = intent['responses']
            
            for pattern in intent['patterns']:
                self.patterns.append((tag, pattern))
        
        # Fit the vectorizer on all patterns
        self.vectorizer.fit([pattern for _, pattern in self.patterns])
    
    def get_response(self, user_input):
        """
        Get a response for the user input by finding the most similar pattern.
        Returns a random response from the matched intent's responses.
        """
        # Vectorize the input
        input_vec = self.vectorizer.transform([user_input])
        
        # Find the most similar pattern
        best_similarity = -1
        best_tag = None
        
        for tag, pattern in self.patterns:
            pattern_vec = self.vectorizer.transform([pattern])
            similarity = cosine_similarity(input_vec, pattern_vec)[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_tag = tag
        
        # If similarity is too low, return a fallback response
        if best_similarity < 0.1:  # Adjust threshold as needed
            return "I'm not sure how to respond to that. Could you try rephrasing?"
        
        # Return a random response from the matched intent
        return random.choice(self.responses[best_tag])

def main():
    print("Chatbot: Hello! I'm a simple chatbot. Type 'quit' to exit.")
    
    # Initialize the chatbot
    chatbot = SimpleChatbot()
    
    # Start the chat loop
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        if not user_input:
            continue
            
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
