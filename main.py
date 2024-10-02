import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Change this to an open-access model
model_name = "facebook/opt-1.3b"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add error handling for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

def get_llama_response(user_question):
    prompt = f"User: {user_question}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    a = 1
    with torch.no_grad():
        a+=1
        print(a)
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    print(a)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()

def main():
    print("Welcome to the AI Chat!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")

    chat_history = []

    while True:
        a = 1
        a += 1
        print(a)
        user_question = (
            f'''
        You are an AI agent that thinks before answering.
Answer in the JSON format. Respond only with the json, no other text.

  "step1": "To solve this problem, i need to do this...",
  "step2": "Based on results from step1, i need to do this...",
  ...
  "stepn": "We can get final result by...",
  "result": "Final result is..."

Your question is:' + {input("You: ")}
'''
        )
        if user_question.lower() in ['quit', 'exit', 'bye']:
            break

        try:
            response = get_llama_response(user_question)
        except Exception as e:
            print(f"Error with LLaMA model: {e}")
            response = "I'm sorry, but I encountered an error. Please try again."

        print("AI:", response)

        chat_history.append({"user": user_question, "ai": response})

    print("Thank you for chatting! Here's a summary of our conversation:")
    for entry in chat_history:
        print(f"You: {entry['user']}")
        print(f"AI: {entry['ai']}")
        print()

if __name__ == "__main__":
    main()
