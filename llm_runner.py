#!/usr/bin/env python3
"""
LLM Runner - Interactive chat interface for Mistral 7B model
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import colorama
from colorama import Fore, Style

try:
    from ctransformers import AutoModelForCausalLM
except ImportError:
    print("Error: ctransformers not installed. Please run: py -3.11 -m pip install -r requirements.txt")
    sys.exit(1)

class LLMRunner:
    def __init__(self, model_path: str):
        """Initialize the LLM runner with the specified model."""
        self.model_path = model_path
        self.llm = None
        self.conversation_history = []
        
    def load_model(self) -> bool:
        """Load the LLM model."""
        try:
            print(f"{Fore.CYAN}Loading model: {self.model_path}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}This may take a few moments...{Style.RESET_ALL}")
            
            # Initialize the model with ctransformers
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                model_type="mistral",
                gpu_layers=0,  # Set to higher number if you have GPU
                threads=4,  # Number of CPU threads
                context_length=4096,  # Context window size
                batch_size=1
            )
            
            print(f"{Fore.GREEN}✓ Model loaded successfully!{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {e}{Style.RESET_ALL}")
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a response from the model."""
        if not self.llm:
            return "Error: Model not loaded"
        
        try:
            # Create the full prompt with conversation history
            full_prompt = self._build_prompt(prompt)
            
            # Generate response
            generated_text = self.llm(
                full_prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>", "\n\n", "[/INST]"],
                stream=False
            )
            
            # Clean up the response
            if isinstance(generated_text, list):
                generated_text = generated_text[0] if generated_text else ""
            
            # Remove the input prompt from the response
            if full_prompt in generated_text:
                generated_text = generated_text[len(full_prompt):].strip()
            
            # Update conversation history
            self.conversation_history.append({"user": prompt, "assistant": generated_text})
            
            return generated_text
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def _build_prompt(self, current_prompt: str) -> str:
        """Build the full prompt with conversation history."""
        if not self.conversation_history:
            return f"<s>[INST] {current_prompt} [/INST]"
        
        # Build conversation context
        context = ""
        for turn in self.conversation_history[-3:]:  # Keep last 3 turns for context
            context += f"<s>[INST] {turn['user']} [/INST] {turn['assistant']} </s>"
        
        context += f"<s>[INST] {current_prompt} [/INST]"
        return context
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        print(f"{Fore.YELLOW}Conversation history cleared.{Style.RESET_ALL}")
    
    def show_help(self):
        """Show available commands."""
        help_text = f"""
{Fore.CYAN}Available Commands:{Style.RESET_ALL}
{Fore.GREEN}help{Style.RESET_ALL}     - Show this help message
{Fore.GREEN}clear{Style.RESET_ALL}    - Clear conversation history
{Fore.GREEN}quit{Style.RESET_ALL}     - Exit the program
{Fore.GREEN}history{Style.RESET_ALL}  - Show conversation history

{Fore.YELLOW}Just type your message to chat with the AI!{Style.RESET_ALL}
        """
        print(help_text)
    
    def show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print(f"{Fore.YELLOW}No conversation history yet.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}Conversation History:{Style.RESET_ALL}")
        for i, turn in enumerate(self.conversation_history, 1):
            print(f"\n{Fore.GREEN}Turn {i}:{Style.RESET_ALL}")
            print(f"{Fore.BLUE}User: {Style.RESET_ALL}{turn['user']}")
            print(f"{Fore.MAGENTA}Assistant: {Style.RESET_ALL}{turn['assistant']}")
        print()

def main():
    """Main function to run the LLM chat interface."""
    # Initialize colorama for cross-platform colored output
    colorama.init()
    
    # Model path
    model_path = "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"{Fore.RED}Error: Model file not found at {model_path}{Style.RESET_ALL}")
        print(f"Please ensure the model file is in the correct location.")
        return
    
    # Initialize and load the model
    runner = LLMRunner(model_path)
    if not runner.load_model():
        return
    
    print(f"\n{Fore.CYAN}=== Mistral 7B LLM Chat Interface ==={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type 'help' for available commands, 'quit' to exit{Style.RESET_ALL}\n")
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}").strip()
            
            # Handle commands
            if user_input.lower() == 'quit':
                print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break
            elif user_input.lower() == 'help':
                runner.show_help()
                continue
            elif user_input.lower() == 'clear':
                runner.clear_history()
                continue
            elif user_input.lower() == 'history':
                runner.show_history()
                continue
            elif not user_input:
                continue
            
            # Generate response
            print(f"{Fore.MAGENTA}Assistant: {Style.RESET_ALL}", end="", flush=True)
            response = runner.generate_response(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted by user. Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    # Cleanup
    colorama.deinit()

if __name__ == "__main__":
    main()
