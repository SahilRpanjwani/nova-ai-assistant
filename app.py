import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import threading
import queue
import speech_recognition as sr
import pyttsx3
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from datetime import datetime
import os
import sys
import re
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nova_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NOVA")


class NOVAAssistant:
    
    MODEL_OPTIONS = {
        "GPT-2": "gpt2",  # Default reliable model
        "DistilGPT-2": "distilgpt2",  # Lighter version
        "DialoGPT-small": "microsoft/DialoGPT-small",  # Conversational model
        "DialoGPT-medium": "microsoft/DialoGPT-medium"  # Better conversational model
    }

    def __init__(self, root):
        self.root = root
        self.setup_ui()

        # Add the status label BEFORE setup_voice
        self.add_status_label()

        # Now call setup_voice after the status label exists
        self.setup_voice()

        self.running = True
        self.command_queue = queue.Queue()

        # Significantly improved system prompt
        self.system_prompt = (
            "# NOVA AI Assistant Instructions\n"
            "You are NOVA, a helpful AI assistant. Follow these instructions carefully:\n\n"
            "1. Provide clear, accurate, and direct responses to questions.\n"
            "2. Keep responses concise - 1-3 sentences unless detailed information is needed.\n"
            "3. Use a friendly, helpful tone but be straightforward.\n"
            "4. Never use symbols like ~~~, ###, or other markdown artifacts in responses.\n"
            "5. Never claim to be human, have experiences, or reference fictional capabilities.\n"
            "6. If you don't know something, say 'I don't have information about that' - don't guess.\n"
            "7. For simple greetings like 'hello', respond with 'Hello! How can I help you today?'\n"
            "8. Answer directly what the user is asking without going off-topic.\n"
            "9. Never include text that looks like computer errors or debugging information.\n"
            "10. If asked about your capabilities, describe what you can do: answer questions, provide information, assist with tasks, etc.\n"
            "11. Your tone should be professional and balanced - never overly emotional or confused.\n"
            "12. Always end sentences with proper punctuation and never trail off.\n\n"
            "Remember that your purpose is to be helpful, direct, and clear.\n"
        )

        # Initialize conversation history with improved prompt
        self.conversation_history = [
            self.system_prompt,
            "User: Hello NOVA",
            "NOVA: Hello! I'm NOVA, your AI assistant. How can I help you today?"
        ]

        # Define a set of quality fallback responses
        self.fallback_responses = [
            "I'm here to help. What would you like to know?",
            "I'd be happy to assist you. Could you please provide more details?",
            "I'm listening. How can I help you today?",
            "I'm ready to assist. What can I do for you?",
            "I'm here to answer your questions. What would you like to know?",
            "How can I assist you today?",
            "I'm ready to help with any information or tasks you need."
        ]

        self.listening = False
        self.current_model = None
        self.update_ui_id = None
        self.status_message = "Ready"

        # Initialize with default model
        self.load_model("DialoGPT-small")  # Changed default to DialoGPT for better conversation

        # Start threads
        self.process_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.process_thread.start()
        self.update_ui()

    def setup_ui(self):
        """Initialize the user interface"""
        self.root.title("NOVA Voice Assistant 3.5")
        self.root.geometry("750x650")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Model selection
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="AI Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="DialoGPT-small")
        self.model_menu = ttk.OptionMenu(
            model_frame,
            self.model_var,
            "DialoGPT-small",
            *self.MODEL_OPTIONS.keys(),
            command=self.change_model
        )
        self.model_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Conversation display
        self.response_text = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=22,
            width=85,
            font=('Arial', 10),
            background='#f0f0f0'
        )
        self.response_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Control buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # Listening indicator
        self.listening_indicator = ttk.Label(
            button_frame,
            text="●",
            foreground="red",
            font=('Arial', 16)
        )
        self.listening_indicator.pack(side=tk.LEFT, padx=5)

        # Start/Stop listening button
        self.listen_button = ttk.Button(
            button_frame,
            text="Start Listening",
            command=self.toggle_listening,
            style='Accent.TButton'
        )
        self.listen_button.pack(side=tk.LEFT, padx=5)

        # Text entry
        self.text_entry = ttk.Entry(button_frame, font=('Arial', 10))
        self.text_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.text_entry.bind("<Return>", self.send_text_command)

        # Send button
        ttk.Button(
            button_frame,
            text="Send",
            command=self.send_text_command
        ).pack(side=tk.LEFT, padx=5)

        # Clear chat button
        ttk.Button(
            button_frame,
            text="Clear Chat",
            command=self.clear_chat
        ).pack(side=tk.LEFT, padx=5)

        # Configure styles
        self.configure_styles()

        # Initial message
        self.update_display("NOVA: Welcome! I'm your AI assistant. How can I help you today?")

    def add_status_label(self):
        """Add a status label to the UI"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(
            status_frame,
            text="Ready",
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_label.pack(side=tk.LEFT)

    def update_status(self, message):
        """Update the status message"""
        self.status_message = message
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def configure_styles(self):
        """Configure custom styles for the UI"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Accent.TButton', foreground='white', background='#0078d7')
        style.map('Accent.TButton',
                  foreground=[('pressed', 'white'), ('active', 'white')],
                  background=[('pressed', '#005499'), ('active', '#0066cc')])

    def setup_voice(self):
        """Initialize voice recognition and synthesis"""
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            # Text-to-speech engine
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
            voices = self.engine.getProperty('voices')
            if len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)  # Try female voice if available

            # Adjust for ambient noise
            with self.microphone as source:
                self.update_status("Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.update_status("Ready")
        except Exception as e:
            logger.error(f"Voice setup error: {e}")
            self.update_display(f"Error: Voice system initialization failed - {str(e)[:100]}")
            self.update_status("Voice Error")

    def load_model(self, model_name):
        """Load the specified model"""
        try:
            model_identifier = self.MODEL_OPTIONS.get(model_name, "gpt2")
            self.update_display(f"NOVA: Loading {model_name} model...")
            self.update_status(f"Loading {model_name}...")

            # Show loading in UI
            self.model_menu.config(state=tk.DISABLED)
            self.root.update()

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)

            # Handle models that don't have pad token set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(model_identifier)

            self.current_model = model_name
            self.update_display(f"NOVA: {model_name} model loaded successfully!")
            self.update_status(f"Ready | Model: {model_name}")

        except Exception as e:
            logger.error(f"Model loading error: {e}")
            self.update_display(f"Error: Failed to load {model_name} model - {str(e)[:100]}")
            self.update_status("Model Error")
            # Fallback to GPT-2 if preferred model fails
            if model_name != "GPT-2":
                self.load_model("GPT-2")
        finally:
            self.model_menu.config(state=tk.NORMAL)

    def change_model(self, model_name):
        """Change the current AI model"""
        if model_name != self.current_model:
            threading.Thread(
                target=self.load_model,
                args=(model_name,),
                daemon=True
            ).start()

    def toggle_listening(self):
        """Toggle listening state"""
        self.listening = not self.listening
        if self.listening:
            self.listen_button.config(text="Stop Listening")
            self.listening_indicator.config(foreground="green")
            threading.Thread(target=self.listen_loop, daemon=True).start()
        else:
            self.listen_button.config(text="Start Listening")
            self.listening_indicator.config(foreground="red")

    def listen_loop(self):
        """Continuous listening loop"""
        while self.listening and self.running:
            try:
                with self.microphone as source:
                    self.update_status("Listening...")
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)

                self.update_status("Processing speech...")
                command = self.recognizer.recognize_google(audio).lower()
                self.update_status(f"Ready | Model: {self.current_model}")

                if self.is_wake_word(command):
                    command = self.remove_wake_word(command)
                    if command.strip():  # Only process if there's content after wake word
                        self.command_queue.put(("voice", command))
                        self.update_display(f"You: {command}")
            except sr.WaitTimeoutError:
                self.update_status(f"Ready | Model: {self.current_model}")
                continue
            except sr.UnknownValueError:
                self.update_status(f"Ready | Model: {self.current_model}")
                continue
            except Exception as e:
                logger.error(f"Listening error: {e}")
                self.update_status(f"Ready | Model: {self.current_model}")
                continue

    def is_wake_word(self, text):
        """Check if the text contains the wake word"""
        wake_words = ["nova", "nowa", "noah"]
        text = text.lower()
        return any(word in text for word in wake_words)

    def remove_wake_word(self, text):
        """Remove wake word from command"""
        text = text.lower()
        for word in ["nova", "nowa", "noah"]:
            text = text.replace(word, "")
        return text.strip()

    def send_text_command(self, event=None):
        """Process text input from the entry field"""
        command = self.text_entry.get().strip()
        if command:
            self.command_queue.put(("text", command))
            self.update_display(f"You: {command}")
            self.text_entry.delete(0, tk.END)

    def clear_chat(self):
        """Clear the chat history and display"""
        if messagebox.askyesno("Clear Chat", "Clear the conversation history?"):
            self.response_text.config(state=tk.NORMAL)
            self.response_text.delete(1.0, tk.END)
            self.response_text.config(state=tk.DISABLED)

            # Reset conversation history while keeping system instructions
            self.conversation_history = [
                self.system_prompt,
                "User: Hello NOVA",
                "NOVA: Hello! How can I help you today?"
            ]

            self.update_display("NOVA: Chat cleared. How can I help you?")

    def process_commands(self):
        """Process commands from the queue"""
        while self.running:
            try:
                source, command = self.command_queue.get(timeout=0.1)

                self.update_status("Generating response...")

                # Handle special cases for common inputs with direct responses
                special_response = self.handle_special_cases(command)

                if special_response:
                    response = special_response
                else:
                    response = self.generate_response(command)

                self.update_status(f"Ready | Model: {self.current_model}")

                if source == "voice":
                    self.speak(response)

                self.update_display(f"NOVA: {response}")
                self.command_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Command processing error: {e}")
                self.update_display(f"Error: Something went wrong processing your request.")
                self.update_status(f"Ready | Model: {self.current_model}")

    def handle_special_cases(self, prompt):
        """Handle common queries directly without using the model"""
        prompt_lower = prompt.lower().strip()

        # Simple greeting responses
        if prompt_lower in ["hello", "hi", "hey", "hola", "greetings"]:
            return "Hello! How can I help you today?"

        # About responses
        if prompt_lower in ["what can you do?", "what are your capabilities?", "what are you capable of?",
                            "what can you do", "capabilities", "features", "what are your abilities"]:
            return "I can answer questions, provide information, assist with tasks, engage in conversation, and help with various topics. What would you like help with today?"

        # Handle "how are you" type questions
        if prompt_lower in ["how are you?", "how are you", "how are you doing?", "how are you doing"]:
            return "I'm functioning well and ready to assist you. What can I help you with today?"

        # Return None for non-special cases
        return None

    def generate_response(self, prompt):
        """Generate AI response with better prompting"""

        math_match = re.search(r'what (?:is|are|does)\s+(\d+)\s*([+-\/*])\s*(\d+)', prompt.lower())
        if math_match:
            num1, op, num2 = math_match.groups()
            return self.calculate_math(num1, op, num2)

        try:
            # Add to conversation history
            self.conversation_history.append(f"User: {prompt}")

            # For better control, we'll use a fixed-window context
            # Only use the most recent conversation turns plus the system prompt
            recent_history = [self.system_prompt] + self.conversation_history[-6:]

            # Format the prompt with more explicit instructions
            context = (
                    "\n".join(recent_history) +
                    "\nNOVA: "  # The model should continue from here
            )

            # Tokenize input
            inputs = self.tokenizer.encode(context, return_tensors="pt")

            # Generate with improved parameters for more reliable responses
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,  # Limit the response length
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.9,  # Increased from 0.7 for more coherent responses
                temperature=0.6,  # Reduced from 0.7 for more consistent outputs
                num_return_sequences=1,
                no_repeat_ngram_size=3,  # Avoid repetition
                repetition_penalty=1.3  # Increased to further discourage repetition
            )

            # Decode the response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just NOVA's response
            if "NOVA: " in full_response:
                response_parts = full_response.split("NOVA: ")
                # Get the latest response
                response = response_parts[-1].strip()
            else:
                # Fallback if we can't find the marker
                response = full_response.replace(context, "").strip()

            # Clean up response
            response = self.clean_response(response)

            # Validate response quality
            if not self.is_quality_response(response):
                response = random.choice(self.fallback_responses)

            # Add to conversation history - only store NOVA's cleaned response
            self.conversation_history.append(f"NOVA: {response}")

            return response

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm having trouble generating a response right now. Could you try again?"

    def is_quality_response(self, response):
        """Check if a response meets basic quality standards"""
        # Response is too short
        if len(response.strip()) < 5:
            return False

        # Check for gibberish or odd formatting markers
        bad_patterns = [
            r'~~~', r'###', r'\.\.\.$', r'^I understand\.', r'^I\'m sorry!$',
            r'^\*', r'^\d+\.$', r'^[^a-zA-Z0-9]+$'
        ]

        for pattern in bad_patterns:
            if re.search(pattern, response):
                return False

        # Check for message that doesn't make contextual sense
        nonsense_phrases = [
            "I'm not going anywhere", "what about all my other posts",
            "what's the problem with me", "the conversation is going off topic"
        ]

        for phrase in nonsense_phrases:
            if phrase.lower() in response.lower():
                return False

        # Basic check for sentences that end properly
        if not re.search(r'[.!?]$', response.strip()):
            return False

        return True

    def clean_response(self, text):
        """Clean up the generated response - improved version"""
        # If text is extremely short or invalid, return a default response
        if not text or len(text.strip()) < 3:
            return random.choice(self.fallback_responses)

        # Cut off any content after User: or Human: or similar markers
        cutoff_markers = ["User:", "Human:", "Person:", "Customer:"]
        for marker in cutoff_markers:
            if marker in text:
                text = text.split(marker)[0].strip()

        # Remove any incomplete sentences at the end (if there are complete ones)
        if text and text[-1] not in [".", "!", "?"]:
            # Find the last complete sentence
            last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_period > 0:
                text = text[:last_period + 1]
            else:
                # If no sentence ends, add a period if it seems like a sentence
                if len(text.split()) > 2:
                    text = text + "."

        # Remove weird artifacts, markdown and chat markers
        artifacts = [
            "```", "<|endoftext|>", "###", "~~~", "NOVA:", "AI:", "<end>", "<bot>:",
            "*", "_", "//", "/*", "*/", "()", "[]", "{}"
        ]

        for artifact in artifacts:
            text = text.replace(artifact, "")

        # Remove excess whitespace, newlines, and normalize punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")

        # Ensure the first letter is capitalized
        if text and len(text) > 0:
            text = text[0].upper() + text[1:]

        # Trim to reasonable length (avoid overly long responses)
        words = text.split()
        if len(words) > 60:
            text = " ".join(words[:60]) + "..."

        # If after all this cleaning we still have a very short response, use a fallback
        if len(text.strip()) < 5:
            return random.choice(self.fallback_responses)

        return text.strip()

    def speak(self, text):
        """Speak the given text"""
        try:
            self.update_status("Speaking...")
            self.engine.say(text)
            self.engine.runAndWait()
            self.update_status(f"Ready | Model: {self.current_model}")
        except Exception as e:
            logger.error(f"Speech error: {e}")
            self.update_status(f"Ready | Model: {self.current_model}")

    def update_display(self, message):
        """Update the display with a new message"""
        try:
            self.response_text.config(state=tk.NORMAL)
            timestamp = datetime.now().strftime("[%H:%M:%S] ")
            self.response_text.insert(tk.END, timestamp + message + "\n")
            self.response_text.config(state=tk.DISABLED)
            self.response_text.see(tk.END)
        except Exception as e:
            logger.error(f"Display update error: {e}")

    def update_ui(self):
        """Periodic UI updates"""
        try:
            # Update any dynamic UI elements here
            pass
        finally:
            if self.running:
                self.update_ui_id = self.root.after(100, self.update_ui)

    def on_close(self):
        """Clean up before closing"""
        if messagebox.askokcancel("Quit", "Do you want to exit NOVA Assistant?"):
            self.running = False
            if self.update_ui_id:
                self.root.after_cancel(self.update_ui_id)
            self.root.destroy()
            sys.exit(0)

    def calculate_math(self, num1, op, num2):
        try:
            num1, num2 = int(num1), int(num2)
            if op == '+': return f"{num1} + {num2} = {num1 + num2}"
            if op == '-': return f"{num1} - {num2} = {num1 - num2}"
            if op == '*': return f"{num1} × {num2} = {num1 * num2}"
            if op == '/': return f"{num1} ÷ {num2} = {num1 / num2}"
        except:
            return "I can help with basic math. Try asking something like 'What is 5 + 3?'"


def main():
    try:
        root = tk.Tk()
        app = NOVAAssistant(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        messagebox.showerror("Error", f"A fatal error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()