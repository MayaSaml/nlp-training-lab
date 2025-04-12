# Health Appointment Scheduling Chatbot

health_appointment_scheduling_chatbot.ipynb notebook demonstrates how to build a chatbot for scheduling health appointments using three different approaches:

1. **Rule-based Logic (Regex):**  
   A simple slot-filling assistant that asks the user for appointment details like doctor type, time, location, etc.

2. **OpenAI Function Calling:**  
   Uses OpenAI's GPT (e.g., `gpt-4-turbo`) with structured function calling to extract information from natural language.

3. **LangChain + Pydantic Output Parsing:**  
   A more modular implementation using LangChain for extraction, memory handling, structured output, and dynamic question generation.

## Try it Out

You can test the chatbot directly in the notebook:
- Type `"start"` to begin the chat.
- The bot will walk you through booking a medical appointment step-by-step.
- Use `"exit"` to end the session.

## Requirements

- `openai`
- `langchain`
- `pydantic`
- `python-dotenv` (optional if using `.env` for your OpenAI key)

Install via:
```bash
pip install openai langchain pydantic python-dotenv
