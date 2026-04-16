import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ==========================================
# Milestone 2: Agentic AI Layer
# ==========================================
# This class makes our project "Smart" by using a Large Language Model (LLM).
# Traditional ML tells us IF someone will leave. 
# This Agent tells us WHY and HOW to stop them.

class ChurnAnalystAgent:
    def __init__(self):
        # Get the API key safely from environment variables
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                self.client = Groq(api_key=api_key)
                # We will prioritize a supported model
                self.model_name = 'llama-3.3-70b-versatile'
                self.active = True
                print(f"✅ AI Agent active using model: {self.model_name}")
            except Exception as e:
                print(f"❌ Failed to start AI Agent: {str(e)}")
                self.active = False
        else:
            print("❌ GROQ_API_KEY not found in environment")
            self.active = False

    def build_context(self, customer_data, feature_importance_text):
        """
        Step 3: Context Builder
        """
        context = f"""
        Customer Data: {json.dumps(customer_data, indent=2)}
        
        Key ML Feature Contributions:
        {feature_importance_text}
        """
        return context

    def get_customer_insight(self, context, model_prediction, rag_context="None"):
        """
        Step 5 & 6: LLM Reasoning and Insight Generation
        """
        if not self.active:
            raise RuntimeError("AI Agent is missing or misconfigured! Check GROQ_API_KEY in .env.")

        # We build a 'Prompt' - this is how we talk to the LLM
        prompt = f"""
        System: You are an expert Customer Success Agent for a Telco company.
        
        Context Setup:
        {context}
        
        Machine Learning Insight:
        The model predicts a churn probability of {model_prediction*100:.1f}%.
        (Higher means more likely to leave).
        
        Knowledge Base (RAG Rules):
        {rag_context}
        
        Task:
        1. Analyze why this specific customer might be unhappy based on the Context Setup.
        2. Provide 3 actionable, custom retention strategies (using the Knowledge Base if applicable).
        3. Suggest a business-friendly explanation of why they are at risk.
        4. Suggest a 'Conversation Starter' for a customer support representative.
        
        Output Style:
        Use clear headings and bullet points. Keep it practical, business-friendly, and easy for a human to read.
        """
        
        try:
            # Generate response from Groq
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                temperature=0.5,
                max_tokens=1024,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error calling Groq API: {str(e)}")

    def search_knowledge_base(self, topic):
        """
        Step 4: RAG Retrieval
        A simple version of RAG (Retrieval Augmented Generation).
        In Milestone 2, retrieval of extra info is key!
        """
        # Hardcoded knowledge for now (Simulated RAG)
        kb = {
            "discounts": "Standard retention discounts are 10% for month-to-month and 20% for fiber users.",
            "competitors": "Competitor X is currently offering free installation, so watch out for price-sensitive users.",
            "support": "Internal data shows that customers with 3+ support tickets in a month are 2x more likely to churn.",
            "retention": "General retention guidelines: Offer a free device upgrade for long-tenured customers. Give a 15% discount for 6 months if they switch to a 1-year contract."
        }
        
        # Searching...
        found_info = kb.get(topic.lower(), "Consult general retention guide for this specific issue.")
        return f"Retrieved Knowledge: {found_info}"

# Created by Antigravity for Milestone 2 extension.
