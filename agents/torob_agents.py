# torob_agents.py
import os, sys
sys.path.append(os.path.abspath(".."))
import httpx
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent, ModelSettings, UsageLimits
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from prompt.prompts import *
from sql.similarity_search_db import similarity_search, find_candidate_shops, similarity_search_cat
from sql.sql_utils import execute_sql
from sql.sql_utils import get_chat_history, get_base_id_and_index
from utils.utils import preprocess_persian
from sql.sql_utils import load_extra_info
from utils.utils import extract_media_type_and_bytes
from pydantic_core import to_jsonable_python
from pydantic_ai.messages import ModelMessagesTypeAdapter  
import json
from pathlib import Path

history_folder = Path("./history")
history_folder.mkdir(parents=True, exist_ok=True)

def save_history(messages, path: str = "chat_history.json"):
    """Serialize agent messages to disk as JSON"""
    as_python_objects = to_jsonable_python(messages)
    Path(path).write_text(
        json.dumps(as_python_objects, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def load_history(path: str = "chat_history.json"):
    """Load messages back into ModelMessages. Return [] if file doesn't exist."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    return ModelMessagesTypeAdapter.validate_python(raw)

# ------------------------
# Load environment
# ------------------------
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

from typing import Optional, List, Tuple
from pydantic import BaseModel

class ConversationResponse(BaseModel):
    """
    Output schema for the conversational product-finding agent.
    
    Includes updated state of constraints/parameters, plus assistant message
    and final candidate (if found).
    """

    # Assistant reply to user
    message: Optional[str] = None

    # Final selection (at most 1 element)
    member_random_keys: Optional[List[str]] = None

    finished: Optional[bool] = False

    # --- Not Changeable parameters ---
    has_warranty: Optional[bool] = None
    score: Optional[float] = None
    city_name: Optional[str] = None
    brand_title: Optional[str] = None
    price_range: Optional[str] = None

    # --- Updateable parameters ---
    product_name: Optional[str] = None
    product_features: Optional[str] = None


class ExtraInfoConversation(BaseModel):
    # --- Not Changeable parameters ---
    has_warranty: Optional[bool] = None
    score: Optional[int] = None
    city_name: Optional[str] = None
    brand_title: Optional[str] = None
    price_range: Optional[Tuple[Optional[int], Optional[int]]] = None

    # --- Updateable parameters ---
    product_name: Optional[str] = None
    product_features: Optional[str] = None

class CompareResponse(BaseModel):
    """Response schema for PRODUCTS_COMPARE queries."""
    message: Optional[str] = None           # justification or reasoning
    base_random_keys: Optional[List[str]] = None  # must contain exactly 1 if chosen

    @field_validator("base_random_keys")
    def ensure_single_key(cls, v):
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("base_random_keys must be a list of strings")
            if len(v) != 1:
                raise ValueError("base_random_keys must contain exactly one random_key")
            if not all(isinstance(x, str) and x.strip() for x in v):
                raise ValueError("base_random_keys must contain non-empty strings")
        return v


from pydantic import BaseModel, field_validator
from typing import Union

class NumericResponse(BaseModel):
    """Response schema for NUMERIC_VALUE queries.
    Must contain only a numeric value parsable to int or float.
    """
    value: Union[int, float]

    @field_validator("value", mode="before")
    def ensure_numeric(cls, v):
        # Accept int or float directly
        if isinstance(v, (int, float)):
            return v
        # Accept string if it can be parsed cleanly
        if isinstance(v, str):
            try:
                num = float(v.strip())
                # If it's a whole number, return int
                if num.is_integer():
                    return int(num)
                return num
            except ValueError:
                pass
        raise ValueError("NumericResponse.value must be parsable to int or float")


class ShoppingResponse(BaseModel):
    """
    The assistant's structured response.

    - message: Explanation or natural-language answer (or null).
    - base_random_keys: list of base product random_key values (or null).
    - member_random_keys: list of member random_key values (or null).
    - finished: Boolean indicating if the answer is definite / final.
    """
    message: Optional[str] = None
    base_random_keys: Optional[List[str]] = None
    member_random_keys: Optional[List[str]] = None
    finished: bool = False

    @field_validator("base_random_keys")
    def validate_base_keys(cls, v):
        if v is not None and (not isinstance(v, list) or not all(isinstance(x, str) for x in v)):
            raise ValueError("base_random_keys must be a list of strings")
        return v

    @field_validator("member_random_keys")
    def validate_member_keys(cls, v):
        if v is not None and (not isinstance(v, list) or not all(isinstance(x, str) for x in v)):
            raise ValueError("member_random_keys must be a list of strings")
        return v


class ClassificationResponse(BaseModel):
    """Output of TorobClassifierAgent."""
    classification: str

    @field_validator("classification")
    def ensure_valid_class(cls, v):
        allowed = {"PRODUCT_SEARCH", "PRODUCT_FEATURE", "NUMERIC_VALUE", "PRODUCTS_COMPARE", "CONVERSATION"}
        if v not in allowed:
            raise ValueError(f"classification must be one of {allowed}")
        return v

class ImageTaskClassificationResponse(BaseModel):
    """Output of TorobClassifierAgent."""
    classification: str

    @field_validator("classification")
    def ensure_valid_class(cls, v):
        allowed = {"IMAGE_TOPIC", "IMAGE_SEARCH"}
        if v not in allowed:
            raise ValueError(f"classification must be one of {allowed}")
        return v


class ImageResponseTopic(BaseModel):
    description: Optional[str] = None
    long_description: Optional[str] = None
    candidates: Optional[List[str]] = None
    main_topic: Optional[str] = None

class ImageResponseSearch(BaseModel):
    description: Optional[str] = None
    long_description: Optional[str] = None
    candidate_names: Optional[List[str]] = None
    candidates: Optional[List[str]] = None
    similarities: Optional[List[float]] = None
    top_candidate: Optional[str] = None
# ------------------------
# Base Class
# ------------------------
class TorobAgentBase:
    def __init__(
        self,
        name: str,
        model_name: str,
        system_prompt: str,
        output_type: Optional[BaseModel] = None,
        tools: Optional[list] = None,
        temperature: float = 0.0001,
        max_tokens: int = 1024,
        examples: Optional[List[str]] = None,
    ):
        self.examples = examples or []

        self.client = OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(
                base_url=BASE_URL,
                api_key=API_KEY,
                http_client=httpx.AsyncClient()
            ),
            settings=ModelSettings(temperature=temperature, max_tokens=max_tokens)
        )
        self.agent = Agent(
            name=name,
            model=self.client,
            system_prompt=system_prompt,
            tools=tools or [],
            output_type=output_type,
        )

    async def run(
        self,
        input_text: str,
        image_b64: str = None,
        usage_limits: Optional[UsageLimits] = None,
        few_shot: int = 0,    # NEW
        **kwargs
    ):
        # Prepend few-shot examples if requested
        if few_shot > 0 and self.examples:
            selected_examples = self.examples[:few_shot]
            input_text +=  "\n\n Here are some examples:\n\n" + "\n\n".join(selected_examples)

        if image_b64:
            image_bytes, media_type = extract_media_type_and_bytes(image_b64)
            user_message = [
                input_text,
                BinaryContent(data=image_bytes, media_type=media_type),
            ]
        else:
            user_message = input_text
        if usage_limits:
            result = await self.agent.run(user_message, 
                                          usage_limits=usage_limits,
                                          **kwargs)
        else:
            result = await self.agent.run(user_message, **kwargs)
        return result, result.output

class TorobClassifierAgent(TorobAgentBase):
    def __init__(self):
        super().__init__(
            name="TorobClassifierAgent",
            model_name=os.getenv("CLASSIFIER_MODEL"),
            system_prompt=input_classification_sys_prompt,
            output_type=ClassificationResponse,
        )

class TorobImageTaskClassifierAgent(TorobAgentBase):
    def __init__(self):
        super().__init__(
            name="TorobImageTaskClassifierAgent",
            model_name=os.getenv("CLASSIFIER_MODEL"),
            system_prompt=img_input_classification_sys_prompt,
            output_type=ImageTaskClassificationResponse,
        )

# ------------------------
# Torob Feature Agent
# ------------------------
class TorobFeatureAgent(TorobAgentBase):
    def __init__(self):
        super().__init__(
            name="TorobFeatureAgent",
            model_name=os.getenv("SHOPPING_MODEL"),
            system_prompt=(
                system_role
                + "\n"
                + SYSTEM_PROMPT_PRODUCT_FEATURE
                + "\n"
                + SIMILARITY_SEARCH_NOTES
                + "\n"
                + SQL_NOTES
                + "\n"
                + ADDITIONAL_NOTES
                + "\nYou have access to the following tools:"
                + "\n"
                + similarity_search_tool + "\n" + execute_query_tool
                + "\nBelow is structure of data in database:"
                + schema_prompt
            ),
            tools=[similarity_search, execute_sql],
            output_type=ShoppingResponse,
        )


# ------------------------
# Torob Compare Agent
# ------------------------
class TorobCompareAgent(TorobAgentBase):
    def __init__(self):
        super().__init__(
            name="TorobCompareAgent",
            model_name=os.getenv("SHOPPING_MODEL"),
            system_prompt=(
                system_role
                + "\n"
                + SYSTEM_PROMPT_PRODUCTS_COMPARE
                + "\n"
                + SIMILARITY_SEARCH_NOTES
                + "\n"
                + SQL_NOTES
                + "\n"
                + ADDITIONAL_NOTES
                + "\nYou have access to the following tools:"
                + "\n"
                + similarity_search_tool + "\n" + execute_query_tool
                + "\nBelow is structure of data in database:"
                + schema_prompt
            ),
            tools=[similarity_search, execute_sql],
            output_type=CompareResponse,
        )


# ------------------------
# Torob Product Search Agent
# ------------------------
class TorobProductSearchAgent(TorobAgentBase):
    def __init__(self):
        super().__init__(
            name="TorobProductSearchAgent",
            model_name=os.getenv("SHOPPING_MODEL"),
            system_prompt=(
                system_role
                + "\n"
                + SYSTEM_PROMPT_PRODUCT_SEARCH
                + "\n"
                + SIMILARITY_SEARCH_NOTES
                + "\nYou have access to the following tools:"
                + "\n"
                + similarity_search_tool
            ),
            tools=[similarity_search],
            output_type=ShoppingResponse,
        )


# ------------------------
# Torob Info Agent
# (for numeric queries like lowest/highest/avg price)
# ------------------------
class TorobInfoAgent(TorobAgentBase):
    def __init__(self):
        super().__init__(
            name="TorobInfoAgent",
            model_name=os.getenv("SHOPPING_MODEL"),
            system_prompt=(
                system_role
                + "\n" +
                SYSTEM_PROMPT_NUMERIC_VALUE 
                + "\n"
                + SIMILARITY_SEARCH_NOTES
                + "\n"
                + SQL_NOTES
                + "\nYou have access to the following tools:"
                + "\n"
                + similarity_search_tool + "\n" + execute_query_tool
                + "\nBelow is structure of data in database:"
                + schema_prompt
            ),
            tools=[similarity_search, execute_sql],
            output_type=NumericResponse,
        )

class TorobConversationAgent(TorobAgentBase):
    def __init__(self):
        super().__init__(
            name="TorobConversationAgent",
            model_name=os.getenv("SHOPPING_MODEL"),
            system_prompt=(
                system_role
                + "\n"
                + SYSTEM_PROMPT_CONVERSATION
                + "\nYou have access to the following tools:"
                + "\n"
                + find_candidate_shops_tool + "\n" + similarity_search_tool 
                + "\n" + execute_query_tool
                + "\nBelow is structure of data in database:"
                + schema_prompt
            ),
            tools=[find_candidate_shops,
                   similarity_search, execute_sql],
            output_type=ConversationResponse,
        )

# ------------------------
# Torob Image Agent
# ------------------------
from pydantic_ai import BinaryContent
import pickle

class TorobImageClassifierAgent(TorobAgentBase):
    def __init__(self):
        # Load category labels from pickle
        # with open("categories_by_level.pkl", "rb") as f:
        #     loaded_levels = pickle.load(f)
        # labels_quotes = [f"Level {lvl}: {cats}" "\n" for lvl, cats in loaded_levels.items()]

        # image_system_prompt = image_label_system_prompt + "\n\nCategories:" + "\n".join(labels_quotes)

        super().__init__(
            name="TorobImageClassifierAgent",
            model_name=os.getenv("IMAGE_MODEL"),
            system_prompt=image_label_system_prompt,
            output_type=ImageResponseTopic,
            tools=[similarity_search_cat],
        )


class TorobImageSearchAgent(TorobAgentBase):
    def __init__(self):
        # Load category labels from pickle
        # with open("categories_by_level.pkl", "rb") as f:
        #     loaded_levels = pickle.load(f)
        # labels_quotes = [f"Level {lvl}: {cats}" "\n" for lvl, cats in loaded_levels.items()]

        # image_system_prompt = image_label_system_prompt + "\n" + "\n".join(labels_quotes)

        super().__init__(
            name="TorobImageSearchAgent",
            model_name=os.getenv("IMAGE_MODEL"),
            system_prompt=(
                image_search_system_prompt
                + "\nYou have access to the following tools:"
                + "\n" + similarity_search_tool
                + "\n" + execute_query_tool
            ),
            output_type=ImageResponseSearch,
            tools=[similarity_search, execute_sql],
        )

class TorobScenarioAgent(TorobAgentBase):
    """
    A single agent that selects the proper Torob agent
    based on uppercase scenario label.
    """

    ALLOWED_SCENARIOS = {
        "PRODUCT_SEARCH",
        "PRODUCT_FEATURE",
        "NUMERIC_VALUE",
        "PRODUCTS_COMPARE",
        "CONVERSATION",
        "IMAGE_TOPIC",
        "IMAGE_SEARCH",
    }

    def __init__(self, scenario: str, examples: Optional[List[str]] = None):
        scenario_upper = scenario.upper()

        # Map scenario to the corresponding agent class
        if scenario_upper == "PRODUCT_FEATURE":
            agent = TorobFeatureAgent()
        elif scenario_upper == "PRODUCTS_COMPARE":
            agent = TorobCompareAgent()
        elif scenario_upper == "PRODUCT_SEARCH":
            agent = TorobProductSearchAgent()
        elif scenario_upper == "NUMERIC_VALUE":
            agent = TorobInfoAgent()
        elif scenario_upper == "CONVERSATION":
            agent = TorobConversationAgent()
        elif scenario_upper == "IMAGE_TOPIC":
            agent = TorobImageClassifierAgent()
        elif scenario_upper == "IMAGE_SEARCH":
            agent = TorobImageSearchAgent()

        # Copy attributes from the selected agent
        self.__dict__.update(agent.__dict__)
        self.examples = examples or []

# ------------------------
# Torob Hybrid Agent
# ------------------------
class TorobHybridAgent(TorobAgentBase):
    def __init__(self):
        # Initialize a dummy agent, we won't use its run directly
        super().__init__(
            name="TorobHybridAgent",
            model_name=os.getenv("SHOPPING_MODEL"),
            system_prompt="Dummy prompt for hybrid agent",
        )

        # Initialize specialized agents
        self.image_agent_classifier = TorobImageTaskClassifierAgent()

    async def run(self, input_dict: dict, usage_limits: Optional[Any] = None, 
                  use_initial_similarity_search: bool = True):
        """
        Run the hybrid agent. Input dict format:
        {
            "messages": [{"type": "text", "content": ...}, {"type": "image", "content": ...}],
            "chat_id": "..."
        }
        """
        try:
            few_shot = 0
            prompt = ""
            # Extract first text and first image if any
            all_texts = [m["content"] for m in input_dict["messages"] if m["type"] == "text"]
            all_images = [m["content"] for m in input_dict["messages"] if m["type"] == "image"]

            instruction = all_texts[0] if all_texts else None
            user_image = all_images[0] if all_images else None

            # --- Step 1: Handle image request ---
            if user_image:
                result, scenario_label = await self.image_agent_classifier.run(
                    input_text=instruction,
                    # image_b64=user_image,
                    usage_limits=usage_limits
                )
                scenario_label = scenario_label.classification
                print(scenario_label)
                scenario_agent = TorobScenarioAgent(scenario_label)
                result, agent_response = await scenario_agent.run(input_text= instruction,
                                                                  image_b64 = user_image,
                                                                  usage_limits=usage_limits, 
                                                                  few_shot=few_shot)
                print(dict(agent_response))
                output_dict = normalize_to_shopping_response(agent_response)
                return result, output_dict

            chat_id = input_dict["chat_id"]
            base_id, chat_index = get_base_id_and_index(chat_id)   # your existing function
            history = get_chat_history(base_id)[-4:]
            info_chat_index = max(1,chat_index-1)
            extra_info = load_extra_info(base_id, info_chat_index)

            # # --- Step 2: Determine scenario ---
            if not history:
                classifier_agent = TorobClassifierAgent()
                _, class_out = await classifier_agent.run(instruction, usage_limits=usage_limits)
                scenario_label = class_out.classification
                print(scenario_label)
            else:
                scenario_label = 'CONVERSATION'
                history_text = "\n".join(
                    [f"Input No.({(i+1)}): {h['message']}\nResponse No.({(i+1)}): {h['response']}" for i,h in enumerate(history)]
                )
                print(history_text)
                prompt += "Conversation history:\n" + history_text + "\n\n"
                if chat_index ==5:
                    prompt += "[IMPORTANT] This is the Fifth turn. Your response is the end of conversation. You must answer the user now definitively.\n"
                        # Add extra info by now
            # message_history = None
            # local_path = ""
            if scenario_label in ['CONVERSATION']:
                # local_path = history_folder  /  f"{base_id}.json"
                # loaded_history = load_history(local_path)
                # message_history =  ModelMessagesTypeAdapter.validate_python(loaded_history)
                extra_info_text = "\n".join([f"{k} = {v}" for k,v in  extra_info.items()])
                prompt += "Previous Parameters:" + "\n\n"  + extra_info_text + "\n\n"

            scenario_agent = TorobScenarioAgent(scenario_label)
            # Step 1: preprocess input
            preprocessed_instruction = preprocess_persian(instruction)

            # Step 2: optionally run similarity search
            similarity_text = ""
            if use_initial_similarity_search and (scenario_label not in ['CONVERSATION']):
                try:
                    candidates = similarity_search(preprocessed_instruction, top_k=5, probes=20)
                    if candidates[0][-1] > 0.7:
                        # candidates is expected to be list[tuple[str, str, float]]
                        rows = []
                        for rk, name, score in candidates:
                            rows.append(f"{rk} -> {name} -> similarity: {score:.4f}")
                        similarity_text = "\n".join(rows)
                        print("Similarity search results:\n", similarity_text)
                except Exception as e:
                    print(f"Similarity search failed: {e}")
            if similarity_text:  
                # Add initial similarity optionally
                prompt += "\n\nInitial Similarity Search Candidates:\n" + similarity_text
                prompt += "\n" + "The initial similarity search results are provided for convenience.\n"

            # Step 3: build prompt for shopping agent
            prompt_prefix = f"Input ({chat_index}): " if scenario_label in ['CONVERSATION'] else "Input: "
            prompt += prompt_prefix + preprocessed_instruction
            # --- Step 3: Run the chosen scenario agent ---
            result, agent_response = await scenario_agent.run(prompt, 
                                                              usage_limits=usage_limits, 
                                                              few_shot=few_shot,
                                                            #   message_history= message_history
                                                              )
            # if scenario_label in ['CONVERSATION']:
            #     current_messages = result.new_messages()
            #     print(current_messages)
            #     save_history(current_messages, local_path)
            # --- Step 4: Normalize output ---
            output_dict = normalize_to_shopping_response(agent_response)
            return result, output_dict

        except Exception as e:
            error_response = ShoppingResponse(
                message=f"-- ERROR: {str(e)}",
                base_random_keys=None,
                member_random_keys=None,
                finished=True,
            )
            return None, dict(error_response)

def normalize_to_shopping_response(output_obj: BaseModel) -> ShoppingResponse:
    """
    Converts any scenario agent output to a ShoppingResponse.
    """
    if isinstance(output_obj, ShoppingResponse):
        final_response = dict(output_obj)

    elif isinstance(output_obj, CompareResponse):
        final_response = ShoppingResponse(
            message=output_obj.message,
            base_random_keys=output_obj.base_random_keys,
            member_random_keys=None,
            finished=True
        )

    elif isinstance(output_obj, NumericResponse):
        final_response = ShoppingResponse(
            message=str(output_obj.value),
            base_random_keys=None,
            member_random_keys=None,
            finished=True
        )

    elif isinstance(output_obj, ImageResponseTopic):
        final_response = ShoppingResponse(
            message=output_obj.main_topic,
            base_random_keys=None,
            member_random_keys=None,
            finished=True
        )
    elif isinstance(output_obj, ImageResponseSearch):
        final_response = ShoppingResponse(
            message=None,
            base_random_keys=[output_obj.top_candidate],
            member_random_keys=None,
            finished=True
        )

    elif isinstance(output_obj, ClassificationResponse):
        final_response = ShoppingResponse(
            message=output_obj.classification,
            base_random_keys=None,
            member_random_keys=None,
            finished=True
        )
    elif isinstance(output_obj, ConversationResponse):
        finished = output_obj.finished
        members_keys = output_obj.member_random_keys if finished else None
        final_response = ShoppingResponse(
            message=output_obj.message,
            base_random_keys=None,
            member_random_keys=members_keys,
            finished=output_obj.finished  # optional, default True
        )

    else:
        # fallback
        final_response = ShoppingResponse(
            message=str(output_obj),
            base_random_keys=None,
            member_random_keys=None,
            finished=True
        )
    final_response = dict(final_response)
    if isinstance(output_obj, ConversationResponse):
        extra_info={
                    "has_warranty": output_obj.has_warranty,
                    "score": output_obj.score,
                    "city_name": output_obj.city_name,
                    "brand_title": output_obj.brand_title,
                    "price_range": output_obj.price_range,
                    "product_name": output_obj.product_name,
                    "product_features": output_obj.product_features,
                    }
        final_response['extra_info'] = extra_info
    return final_response