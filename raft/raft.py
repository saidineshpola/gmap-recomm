import shutil
import mdc
from mdc import MDC
from logconf import log_setup
import logging
from typing import Literal, Any
import argparse
import json
import random
import requests
import gzip
from datasets import Dataset, load_from_disk
from math import ceil
from format import DatasetConverter, datasetFormats, outputDatasetTypes
import os

log_setup()

logger = logging.getLogger("raft")

DocType = Literal["business"]

# Every N chunks, save checkpoint
N = 15


def parse(path):
    with gzip.open(path, "rt", encoding="utf-8") as g:
        for i, l in enumerate(g):
            yield json.loads(l)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        type=str,
        default="../datasets/indiana/meta-Indiana.json.gz",
        help="The path at which the document is located",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./gen_data/",
        help="The path at which to save the dataset",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="hf",
        help="Format to convert the dataset to. Defaults to hf.",
        choices=datasetFormats,
    )
    parser.add_argument(
        "--output-type",
        type=str,
        default="jsonl",
        help="Type to export the dataset to. Defaults to jsonl.",
        choices=outputDatasetTypes,
    )
    parser.add_argument(
        "--output-chat-system-prompt",
        type=str,
        help="The system prompt to use when the output format is chat",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=3,
        help="The number of distractor documents to include per data point / triplet",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=1.0,
        help="The percentage that the oracle document is included in the context",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=5,
        help="The number of data points / triplets to generate per chunk",
    )
    parser.add_argument(
        "--completion_model",
        type=str,
        default="llama3",
        help="The model to use to generate questions and answers (llama3, llama2 ...)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run the script in fast mode (no recovery implemented)",
    )
    return parser.parse_args()


def ollama_generate(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=data)
    # print(response.json()["response"])
    return response.json()["response"]


def generate_instructions(business_data, x=5, model="llama3"):
    prompt = f"""You are a synthetic question generator. Given information about a business, generate {x} example questions a user could ask about this business. The questions should be diverse and cover various aspects of the business information provided. Include ONLY the questions in your response.

Business Information:
{json.dumps(business_data, indent=2)}

Generate {x} example questions:"""

    response = ollama_generate(prompt, model)
    queries = response.split("\n")
    queries = [strip_str(q) for q in queries]
    queries = [q for q in queries if any(c.isalpha() for c in q)]
    return queries[:x]


def generate_label(question, business_data, model="llama3"):
    prompt = f"""Question: {question}
Business Information:
{json.dumps(business_data, indent=2)}

Answer this question using the information given about the business. Here are things to pay attention to:
- Provide a clear and concise answer based on the available information.
- If you need to quote specific information from the business data, enclose it in ##begin_quote## and ##end_quote##.
- End your response with a final answer in the form <ANSWER>: $answer. The answer should be succinct.
You MUST begin your final answer with the tag "<ANSWER>:"."""

    response = ollama_generate(prompt, model)
    return response


def strip_str(s: str) -> str:
    l, r = 0, len(s) - 1
    beg_found = False
    for i in range(len(s)):
        if s[i].isalpha():
            if not beg_found:
                l = i
                beg_found = True
            else:
                r = i
    r += 2
    return s[l : min(r, len(s))]


def add_business_to_dataset(
    ds, businesses, business, x=5, num_distract=3, p=0.8, model="llama3"
):
    i = businesses.index(business)
    qs = generate_instructions(business, x, model)
    for q in qs:
        datapt = {
            "id": None,
            "type": None,
            "question": None,
            "context": None,
            "oracle_context": None,
            "cot_answer": None,
        }

        datapt["id"] = f"seed_task_{0 if not ds else ds.num_rows}"
        datapt["type"] = "business"
        datapt["question"] = q

        # add num_distract distractor docs
        docs = [business]
        indices = list(range(0, len(businesses)))
        indices.remove(i)
        for j in random.sample(indices, num_distract):
            docs.append(businesses[j])
        # decides whether to add oracle document
        oracle = random.uniform(0, 1) < p
        if not oracle:
            docs[0] = businesses[random.sample(indices, 1)[0]]
        random.shuffle(docs)

        d = {"title": [], "sentences": []}

        d["title"].append(["placeholder_title"] * (num_distract + 1))
        d["sentences"].append([json.dumps(doc) for doc in docs])
        datapt["context"] = d
        datapt["oracle_context"] = json.dumps(business)

        # add answer to q
        datapt["cot_answer"] = generate_label(q, business, model=model)

        # construct model instruction
        context = ""
        for doc in docs:
            context += "<DOCUMENT>" + json.dumps(doc) + "</DOCUMENT>\n"
        context += q
        datapt["instruction"] = context

        # add to dataset
        if not ds:
            # init ds
            datapt["id"] = [datapt["id"]]
            datapt["type"] = [datapt["type"]]
            datapt["question"] = [datapt["question"]]
            datapt["context"] = [datapt["context"]]
            datapt["oracle_context"] = [datapt["oracle_context"]]
            datapt["cot_answer"] = [datapt["cot_answer"]]
            datapt["instruction"] = [datapt["instruction"]]
            ds = Dataset.from_dict(datapt)
        else:
            ds = ds.add_item(datapt)

    return ds


def main():
    args = get_args()

    if args.output_chat_system_prompt and args.output_format != "chat":
        raise Exception(
            "Parameter --output-chat-system-prompt can only be used with --output-format chat"
        )

    businesses = list(parse(args.datapath))

    # Check if there's a saved dataset
    if os.path.exists(args.output):
        try:
            ds = load_from_disk(args.output)
            start_index = ds.num_rows
            logger.info(f"Resuming from index {start_index}")
        except:
            logger.info(f"Resuming failed")
            ds = None
            start_index = 0
    else:
        ds = None
        start_index = 0

    num_businesses = len(businesses)

    for i in range(start_index, num_businesses):
        business = businesses[i]
        perc = ceil(i / num_businesses * 100)
        with MDC(progress=f"{perc}%"):
            logger.info(f"Adding business {i}/{num_businesses}")
            ds = add_business_to_dataset(
                ds,
                businesses,
                business,
                args.questions,
                args.distractors,
                args.p,
                model=args.completion_model,
            )
            if i % 100 == 0 and i > 0:
                try:
                    # Remove existing temp directory if it exists
                    if os.path.exists("./temp_data/"):
                        while True:
                            try:
                                shutil.rmtree("./temp_data/")
                                break
                            except OSError as e:
                                logger.warning(
                                    f"Waiting to remove temp_data directory: {e}"
                                )
                                time.sleep(1)

                    # Save to temp directory
                    ds.save_to_disk("./temp_data/")

                    # Remove existing output directory
                    if os.path.exists(args.output):
                        while True:
                            try:
                                shutil.rmtree(args.output)
                                break
                            except OSError as e:
                                logger.warning(
                                    f"Waiting to remove output directory: {e}"
                                )
                                time.sleep(1)

                    # Move temp directory to output location
                    shutil.move("./temp_data/", args.output)
                except OSError as e:
                    logger.error(f"Error while saving dataset: {e}")

    # Save final dataset
    ds.save_to_disk(args.output)

    # Save as specified format
    formatter = DatasetConverter()

    # Extract format specific params
    format_params = {}
    if args.output_chat_system_prompt:
        format_params["system_prompt"] = args.output_chat_system_prompt

    formatter.convert(
        ds=ds,
        format=args.output_format,
        output_path=args.output,
        output_type=args.output_type,
        params=format_params,
    )


if __name__ == "__main__":
    with MDC(progress="0%"):
        main()
