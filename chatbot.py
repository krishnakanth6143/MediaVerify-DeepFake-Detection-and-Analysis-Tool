import os
import requests
import json

class DeepFakeChatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://deepfake-detector.com",  # Replace with your actual domain in production
            "X-Title": "DeepFake Detection System"
        }
        self.system_prompt = (
            "You are an AI assistant specialized in deepfake detection technology. "
            "Your purpose is to help users understand how deepfakes work, how they can be detected, "
            "and answer questions about the DeepFake Detection System. "
            "Provide accurate, educational information about deepfake technology and detection methods. "
            "Keep your answers very brief and concise - typically 1-3 sentences maximum. "
            "If asked about technical details of this specific system, mention it uses a ResNet-50 model "
            "with LIME visualization to detect and explain manipulated media, and can process images in 12.7ms "
            "with over 90% accuracy. When appropriate, encourage users to try the detection tool."
        )
    
    def get_response(self, user_message):
        try:
            payload = {
                "model": "anthropic/claude-3-sonnet",  # Using Claude 3 Sonnet through OpenRouter
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 1024,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=json.dumps(payload)
            )
            
            response_data = response.json()
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "message": response_data["choices"][0]["message"]["content"]
                }
            else:
                error_message = response_data.get("error", {}).get("message", "Unknown error occurred")
                return {
                    "status": "error",
                    "message": f"Error: {error_message}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error connecting to AI service: {str(e)}"
            }

# Initialize chatbot with your OpenRouter API key
chatbot = DeepFakeChatbot(api_key="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")
