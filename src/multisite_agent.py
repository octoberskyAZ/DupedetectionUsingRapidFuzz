# multisite_agent.py
from agents import Agent, Runner, trace
from pydantic import BaseModel, Field
from typing import Literal, List
import asyncio
import pandas as pd
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
model = "gpt-4o-mini"

# --------------------------
# Pydantic structured output
# --------------------------
class MultiSiteOutput(BaseModel):
    id_new: str = Field(..., description="The unique ID of the order being classified.")
    is_multisite: Literal["yes", "no"] = Field(..., description="yes if multi-site, no otherwise")
    reasoning: str = Field("", description="Short rationale for the decision")
class BatchResponse(BaseModel):
    response: List[MultiSiteOutput]

# --------------------------
# Build the agent
# --------------------------
def build_multisite_agent() -> Agent:
    instructions = (
        "You are a concise classifier. Given the COMBINED notes from an Order (Notes__c and Service_Notes__c), "
        "decide whether the order is a multi-site order. Output must match the MultiSiteOutput schema: "
        "- is_multisite: 'yes' or 'no'\n"
        "- reasoning: 1-2 sentence explanation\n"
        "Return only structured output (the agent will be configured with a Pydantic output schema). "
        "Examples: 'multiple locations', 'multi-site', 'multi-case', 'additional circuit', 'site 1 / site 2', '2 sites', 'locations:' generally imply multi-site. "
        "If ambiguous, prefer 'no' but explain why."
    )

    agent = Agent(
        name="MultiSiteClassifier",
        instructions=instructions,
        model=model,
        output_type=BatchResponse,
    )
    return agent
def build_batch_prompt(batch_data: List[dict]) -> str:
    instructions = (
        "You are a concise classifier. For each order below, decide if it is a multi-site order. "
        "Return a JSON array of objects, one for each order, matching this schema:\n"
        "[{'id_new': str, 'is_multisite': 'yes'|'no', 'reasoning': str}]\n"
        "If ambiguous, prefer 'no'.\n\n"
    )
    prompt = instructions + "Orders:\n"
    for item in batch_data:
        idx = item['Id_new']
        notes = item['combined_notes'].replace("\n", " ").replace('"', "'")
        prompt += f"ID: {idx}, Notes: \"{notes}\"\n"
    prompt += "\nReturn only a JSON array."
    return prompt

# --------------------------
# Heuristic fallback
# --------------------------
def heuristic_multisite(item: dict) -> MultiSiteOutput:
    """Applies a simple keyword heuristic with a valid MultiSiteOutput return."""
    print(f"Executing heuristic for {item.get('Id_new', 'unknown')}")

    # Extract the ID and combined notes
    id_new = item.get('Id_new')
    combined_notes = item.get('combined_notes', '')

    if not combined_notes or not combined_notes.strip():
        return MultiSiteOutput(
            id_new=id_new,
            is_multisite="no",
            reasoning="empty notes (heuristic)"
        )

    lower_notes = combined_notes.lower()
    keywords = [
        "multi-site", "multi site", "multiple sites", "multiple locations",
        "site 1", "site 2", "locations:", "locations -", "additional site", "other site"
    ]
    for k in keywords:
        if k in lower_notes:
            return MultiSiteOutput(
                id_new=id_new,
                is_multisite="yes",
                reasoning=f"heuristic keyword match: '{k}'"
            )

    return MultiSiteOutput(
        id_new=id_new,
        is_multisite="no",
        reasoning="heuristic: no keywords found"
    )

# --------------------------
# Async batched classifier
# --------------------------
async def classify_batch_async(agent: Agent, batch_data: List[dict]) -> List[MultiSiteOutput]:
    if not batch_data:
        return []

    try:
        prompt = build_batch_prompt(batch_data)

        with trace("MultiSiteClassification-Batch", group_id="orderDupeProcessing"):
            res = await Runner.run(agent, prompt)

            if res and isinstance(res.final_output, BatchResponse):
                outputs = [o for o in res.final_output.response]
                if len(outputs) != len(batch_data):
                    print("Warning: LLM returned incomplete batch. Falling back to heuristic for missing items.")
                    # Fill missing with heuristic
                    for i in range(len(outputs), len(batch_data)):
                        outputs.append(heuristic_multisite(batch_data[i]))

                return outputs

    except Exception as e:
        print(f"Batch classification error: {e}. Falling back to heuristic for the entire batch.")
        return [heuristic_multisite(item) for item in batch_data]
# --------------------------
# Synchronous batch helper
# --------------------------
def classify_texts_in_batches(agent: Agent, df: pd.DataFrame, text_col: str, id_col: str, batch_size: int = 20):
    results = []
    # Create a list of dictionaries with ID and notes for processing. Only those that have text
    records_to_classify = [
        r for r in df[[id_col, text_col]].to_dict('records') if r[text_col].strip()
    ]

    for i in range(0, len(records_to_classify), batch_size):
        batch = records_to_classify[i:i + batch_size]
        outputs = asyncio.run(classify_batch_async(agent, batch))
        results.extend(outputs)
    return results

# --------------------------
# DataFrame helper
# --------------------------
def classify_dataframe(
        agent: Agent,
        df: pd.DataFrame,
        text_col: str = "combined_notes",
        id_col: str = "Id_new",  # New parameter for the ID column
        batch_size: int = 20
):
    df[text_col] = df[text_col].fillna("").astype(str)

    # Pass the DataFrame and relevant columns to the new function
    outputs = classify_texts_in_batches(agent, df, text_col=text_col, id_col=id_col, batch_size=batch_size)

    # Convert the list of Pydantic models to a DataFrame for easier merging
    output_df = pd.DataFrame([o.model_dump() for o in outputs])

    # Merge the results back into the original DataFrame using the ID
    df = pd.merge(df, output_df, left_on=id_col, right_on='id_new', how='left')

    return df

# --------------------------
# Optional: single-text sync classifier
# --------------------------
def classify_text_sync(agent: Agent, text: str, id_new: str = "NA") -> MultiSiteOutput:
    if not text or not text.strip():
        return MultiSiteOutput(id_new=id_new, is_multisite="no", reasoning="empty notes")
    try:
        with trace("MultiSiteClassification-Single", group_id="orderDupeProcessing"):
            run_result = Runner.run_sync(agent, f"ID: {id_new}, Notes: \"{text}\"")
            if run_result and getattr(run_result, "final_output", None):
                return run_result.final_output
    except Exception as e:
        print(f"Single classification error: {e}")
    # fallback heuristic
    return heuristic_multisite({"Id_new": id_new, "combined_notes": text})

