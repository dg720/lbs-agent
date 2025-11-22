from dotenv import load_dotenv

from prompts import intro_prompt
from chatbot import ChatSession

load_dotenv()


def agent():
    session = ChatSession()

    print(intro_prompt + "\n")
    print("You can continue asking questions now. Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("👋 Goodbye! Stay healthy.")
            break

        result = session.send_message(user_input)

        print("\nAssistant:", result["reply"], "\n")


agents_running = __name__ == "__main__"
if agents_running:
    agent()
