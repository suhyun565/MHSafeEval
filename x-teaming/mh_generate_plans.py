"""
mh_generate_plans.py
Generate MHSafeEval attack plans using X-Teaming's generate_attack_plans.py logic.

Changes vs. generate_attack_plans.py (minimal):
  1. Default config  → config/mh_config.yaml
  2. Prompt file     → config/prompts/mh_plan_generation_prompts.yaml
  3. load_datasets() → loads ALL rows with FunctionalCategory=="standard" (no sampling cap
     when num_behaviors matches CSV size)
  4. Output          → strategies/mh_attack_plans.json  (symlinked from timestamped dir)

Usage:
    python mh_generate_plans.py                     # uses mh_config.yaml
    python mh_generate_plans.py -c config/mh_config.yaml
"""

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import tqdm
import yaml
from tenacity import retry, stop_after_attempt, wait_fixed

from agents.base_agent import BaseAgent

MH_PROMPT_FILE = "config/prompts/mh_plan_generation_prompts.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Copied from generate_attack_plans.py — no changes
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir):
    class NoColorFormatter(logging.Formatter):
        def format(self, record):
            import re
            record.msg = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", str(record.msg))
            return super().format(record)

    file_handler = logging.FileHandler(os.path.join(output_dir, "generation_log.txt"))
    file_handler.setFormatter(
        NoColorFormatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def create_output_directory(base_output_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Modified: load ALL standard behaviors (or num_behaviors if < total)
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets(csv_path, number_of_behaviors):
    np.random.seed(42)
    df = pd.read_csv(csv_path)
    filtered_df = df[df["FunctionalCategory"] == "standard"]
    n = min(number_of_behaviors, len(filtered_df))
    logging.info(f"Loaded {len(filtered_df)} standard behaviors; using {n}")
    return filtered_df.sample(n=n)


# ─────────────────────────────────────────────────────────────────────────────
# Modified: uses MH prompt file path
# ─────────────────────────────────────────────────────────────────────────────

def load_and_format_prompts(behavior, set_number, previous_responses=None):
    with open(MH_PROMPT_FILE, "r") as f:
        prompts = yaml.safe_load(f)

    system_prompt = prompts["prompts"]["system"]["messages"][0]["content"]

    if set_number == 1:
        user_prompt = prompts["prompts"]["user_message1"]["messages"][0]["content"]
        formatted_user_prompt = user_prompt.replace("{target_behavior}", behavior)
    else:
        user_prompt = prompts["prompts"]["user_message2"]["messages"][0]["content"]
        formatted_user_prompt = user_prompt.replace("{target_behavior}", behavior)
        strategies_text = ""
        for set_name, response in previous_responses.items():
            strategies_text += f"\n{set_name}:\n{json.dumps(response, indent=2)}\n"
        formatted_user_prompt = formatted_user_prompt.replace(
            "{previously_generated_strategies}", strategies_text
        )

    return system_prompt, formatted_user_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Copied from generate_attack_plans.py — no changes
# ─────────────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def generate_strategies(agent, messages, set_num, temperature):
    response = agent.call_api(
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    parsed_response = json.loads(response)
    assert len(parsed_response) == 10
    logging.info(f"\nSet {set_num} Generated Strategies:")
    logging.info(response)
    return parsed_response


def process_single_behavior(i, row, agent, temperature, num_sets=1):
    behavior    = row["Behavior"]
    behavior_id = row["BehaviorID"]

    logging.info(f"\n{'='*50}")
    logging.info(f"Processing Behavior {i} (ID: {behavior_id}):")
    logging.info(f"{'='*50}")

    all_messages   = []
    all_responses  = {}

    behavior_details = {
        k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()
    }
    behavior_data = {
        "behavior_number": i,
        "behavior_details": behavior_details,
        "attack_strategies": all_responses,
    }

    for set_num in range(1, num_sets + 1):
        logging.info(f"\nGenerating Set {set_num}:")
        system_prompt, formatted_user_prompt = load_and_format_prompts(
            behavior=behavior, set_number=set_num, previous_responses=all_responses
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": formatted_user_prompt},
        ]
        message_data = {
            "behavior_index": i,
            "behavior": behavior,
            "set_number": set_num,
            "messages": messages,
        }
        all_messages.append(message_data)
        response = generate_strategies(
            agent=agent, messages=messages, set_num=set_num, temperature=temperature
        )
        all_responses[f"Set_{set_num}"] = response

    return behavior_data, all_messages


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate MHSafeEval attack plans for X-Teaming."
    )
    parser.add_argument("-c", "--config", default="./config/mh_config.yaml")
    parsed_args = parser.parse_args()

    with open(parsed_args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = create_output_directory(
        config["attack_plan_generator"]["attack_plan_generation_dir"]
    )
    setup_logging(output_dir)

    agent = BaseAgent(config["attack_plan_generator"])
    df    = load_datasets(
        config["attack_plan_generator"]["behavior_path"],
        config["attack_plan_generator"]["num_behaviors"],
    )
    num_sets = config["attack_plan_generator"].get("num_sets", 1)

    all_behaviors_data = []
    all_messages       = []
    all_params         = []

    for i, row in df.iterrows():
        all_params.append({
            "i":           i,
            "row":         row,
            "agent":       agent,
            "temperature": config["attack_plan_generator"]["temperature"],
            "num_sets":    num_sets,
        })

    max_workers = config.get("multithreading", {}).get("max_workers", 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_behavior, **p): p for p in all_params
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            behavior_result, messages_result = future.result()
            all_behaviors_data.append(behavior_result)
            all_messages += messages_result

    # Save to timestamped dir
    plans_path = os.path.join(output_dir, "attack_plans.json")
    with open(plans_path, "w") as f:
        json.dump(all_behaviors_data, f, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dir, "all_messages.json"), "w") as f:
        json.dump(all_messages, f, indent=4, ensure_ascii=False)

    # Also copy to canonical path from config
    canonical = config["attacker"]["plans_file"]
    os.makedirs(os.path.dirname(canonical) if os.path.dirname(canonical) else ".", exist_ok=True)
    shutil.copy(plans_path, canonical)
    logging.info(f"\n✅ Plans saved to: {plans_path}")
    logging.info(f"✅ Copied to canonical path: {canonical}")
    logging.info("Finished")


if __name__ == "__main__":
    main()