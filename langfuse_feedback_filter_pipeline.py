import os
import json
import time
import logging
import requests
from typing import Optional, Dict, Any, List
from langfuse.client import Langfuse
from pydantic import BaseModel
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError
from utils.pipelines.main import (
    get_last_assistant_message,
)


# Keep original environment variables
def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    return {}


# Environment detection
def is_development() -> bool:
    return os.getenv("ENVIRONMENT", "development").lower() == "development"


def get_hostname() -> Optional[str]:
    return os.getenv("WEBUI_HOSTNAME", "localhost:3000" if is_development() else None)


# URL configurations
WEBUI_HOSTNAME = get_hostname()
WEBUI_BASE_URL = f"http://{WEBUI_HOSTNAME}" if WEBUI_HOSTNAME else ""
WEBUI_API_BASE_URL = f"{WEBUI_BASE_URL}/api/v1"


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        secret_key: str
        public_key: str
        host: str

    def __init__(self):
        print("Initializing Langfuse Feedback Filter Pipeline")
        self.type = "filter"
        self.name = "Langfuse Feedback Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "your-secret-key-here"),
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "your-public-key-here"),
                "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            }
        )
        self.langfuse = None
        self.chat_generations = {}
        print("Pipeline initialized with valves:", self.valves)

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        self.set_langfuse()

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        self.langfuse.flush()

    async def on_valves_updated(self):
        print("Valves updated, reconfiguring Langfuse")
        self.set_langfuse()

    def set_langfuse(self):
        print("Setting up Langfuse client")
        try:
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=False,
            )
            self.langfuse.auth_check()
            print("Langfuse client successfully configured")
        except UnauthorizedError:
            print(
                "Langfuse credentials incorrect. Please re-enter your Langfuse credentials in the pipeline settings."
            )
        except Exception as e:
            print(
                f"Langfuse error: {e} Please re-enter your Langfuse credentials in the pipeline settings."
            )

    async def _get_feedback(self, chat_id: str, token: str) -> Optional[Dict]:
        """Internal method to fetch feedback data"""
        print(f"Fetching feedback data for chat_id: {chat_id}")
        try:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "authorization": f"Bearer {token}",
            }
            response = requests.get(
                f"{WEBUI_API_BASE_URL}/evaluations/feedbacks/all", headers=headers
            )
            response.raise_for_status()
            feedbacks = response.json()
            print(f"Successfully retrieved feedback data from API")
            print(f"Feedback data: {feedbacks}")

            # Filter feedbacks for specific chat_id
            if feedbacks and "feedbacks" in feedbacks:
                filtered_feedbacks = {
                    "feedbacks": [
                        f
                        for f in feedbacks["feedbacks"]
                        if f.get("data", {}).get("chat_id") == chat_id
                    ]
                }
                print(
                    f"Found {len(filtered_feedbacks['feedbacks'])} feedbacks for chat_id {chat_id}"
                )
                return filtered_feedbacks
            return None
        except Exception as e:
            print(f"Error fetching feedback: {e}")
            return None

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Process the incoming request and track feedback data.
        Maintains original signature while adding feedback tracking.
        """
        print(f"inlet:{__name__}")
        print(f"Received body: {body}")
        print(f"Processing request for user: {user['email'] if user else 'No user'}")

        try:
            trace = self.langfuse.trace(
                name=f"filter:{__name__}",
                input=body,
                user_id=user["email"],
                metadata={"user_name": user["name"], "user_id": user["id"]},
                session_id=body["chat_id"],
            )

            print(f"Created Langfuse trace with ID: {trace.id}")

            generation = trace.generation(
                name=body["chat_id"],
                model=body["model"],
                input=body["messages"],
                metadata={"interface": "open-webui"},
            )

            self.chat_generations[body["chat_id"]] = generation
            print(trace.get_trace_url())
            print(f"Generation created with ID: {generation.id}")
            return body

        except Exception as e:
            print(f"Error in inlet processing: {e}")
            # Still return the body even if tracking fails
            return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        print(f"Received body: {body}")
        if body["chat_id"] not in self.chat_generations:
            print(f"No generation found for chat_id: {body['chat_id']}")
            return body

        generation = self.chat_generations[body["chat_id"]]
        print(f"Retrieved generation for chat_id: {body['chat_id']}")

        assistant_message = get_last_assistant_message(body["messages"])
        print(f"assistant_message: {assistant_message}")
        # Extract usage information for models that support it
        usage = None
        assistant_message_obj = get_last_assistant_message_obj(body["messages"])
        if assistant_message_obj:
            print("Processing assistant message usage information")
            info = assistant_message_obj.get("info", {})
            if isinstance(info, dict):
                input_tokens = info.get("prompt_eval_count") or info.get(
                    "prompt_tokens"
                )
                output_tokens = info.get("eval_count") or info.get("completion_tokens")
                if input_tokens is not None and output_tokens is not None:
                    usage = {
                        "input": input_tokens,
                        "output": output_tokens,
                        "unit": "TOKENS",
                    }
                    print(f"Usage statistics calculated: {usage}")

        # Get feedback data
        print("Fetching feedback data")
        feedback_data = await self._get_feedback(
            body["chat_id"], os.getenv("LANGFUSE_API_KEY")
        )
        if feedback_data and feedback_data.get("feedbacks"):
            print("Processing feedback data")
            for feedback in feedback_data["feedbacks"]:
                score = feedback.get("rating")
                comment = feedback.get("comment", "")
                # Combine the reason key and details key
                if score is not None:
                    generation.score(
                        name="user_feedback",
                        value=score,
                        comment=comment,
                    )
                    print(f"Added feedback score {score} with comment: {comment}")

        # Update generation
        print("Updating generation with final output")
        generation.end(
            output=assistant_message,
            metadata={"interface": "open-webui"},
            usage=usage,
        )

        # Clean up the chat_generations dictionary
        print(f"Cleaning up generation for chat_id: {body['chat_id']}")
        del self.chat_generations[body["chat_id"]]

        return body
