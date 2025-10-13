class AgentComms:
    def broadcast(self, message):
        print(f"Broadcasting message: {message}")

    def send_message(self, sender, receiver, message):
        print(f"Sending message from {sender} to {receiver}: {message}")
