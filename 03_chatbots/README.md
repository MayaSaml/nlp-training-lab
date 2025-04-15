# Chatbots for Structured Information Extraction

This folder contains interactive chatbot notebooks that demonstrate how to extract structured data from natural language conversations using different approaches and frameworks.

---

## 🤖 1. Health Appointment Scheduling Chatbot

**Notebook:** `health_appointment_scheduling_chatbot.ipynb`  
This chatbot helps users schedule health appointments by extracting relevant information like doctor type, date, and location.

### Key Approaches:
- **Rule-based Logic (Regex):**  
  Slot-filling using simple logic and regex.
- **OpenAI Function Calling:**  
  Leverages GPT with function calling to extract structured fields.
- **LangChain + Pydantic:**  
  Uses LangChain and Pydantic for modular, dynamic slot filling and memory.

---

## 🏠 2. Real Estate Search Chatbot

**Notebook:** `real_estate_chatbot.ipynb`  
This assistant helps users find their ideal property to **buy or rent**, capturing detailed requirements such as:

- Budget (price range)
- Location (city, area, country)
- Number of bedrooms/bathrooms
- Property type (apartment, house, villa, etc.)
- Amenities (e.g., pool, garden, garage, elevator)
- Preferences like size, view, style, proximity, and more

### Features:
- Built using **LangChain + OpenAI**
- Structured schema with **Pydantic models**
- Intelligent range handling (e.g., `2-4 bedrooms`, `up to €500K`)
- Dynamic follow-up questions to collect missing info
- Amenity extraction with keyword and model-based logic

---

## Try It Out

You can run and test the chatbots directly in the notebook:
- Type `"start"` to begin the chat.
- The bot will guide you step-by-step.
- Type `"exit"` to end the session.

---

## Requirements

Install the required libraries:

```bash
pip install openai langchain pydantic python-dotenv
