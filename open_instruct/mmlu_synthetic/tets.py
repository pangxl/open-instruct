import asyncio
import json
import re
import random
import logging
from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from difflib import SequenceMatcher


# Custom logger to filter out unwanted messages
class CustomFilter(logging.Filter):
    def filter(self, record):
        return "HTTP Request:" not in record.getMessage()


# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.addFilter(CustomFilter())

# Suppress OpenAI's HTTP request logging
logging.getLogger("openai").setLevel(logging.WARNING)


@dataclass
class LLMGenerationConfig:
    num_completions: int = 50
    model: str = "gpt-4"
    max_parallel_requests: int = 50
    max_retries: int = 3
    retry_delay: float = 5.0


@dataclass
class GenerationArgs:
    temperature: float = 0.9
    max_tokens: int = 500
    top_p: float = 0.95
    examples_per_subject: int = 500
    few_shot_examples: int = 10
    similarity_threshold: float = 0.8


class LLMProcessor:
    def __init__(self, config: LLMGenerationConfig):
        self.config = config
        self.async_client = AsyncOpenAI()
        self.generated_questions = set()

    def format_example(self, sample: dict) -> str:
        return f"""
Question: {sample['question']}
A: {sample['choices'][0]}
B: {sample['choices'][1]}
C: {sample['choices'][2]}
D: {sample['choices'][3]}
Correct answer: {sample['choices'][sample['answer']]}
"""

    async def generate_question_batch(self, subject: str, examples: List[dict], gen_args: GenerationArgs,
                                      batch_size: int):
        for attempt in range(self.config.max_retries):
            try:
                few_shot_examples = random.sample(examples, k=gen_args.few_shot_examples)
                examples_str = "\n".join([self.format_example(example) for example in few_shot_examples])

                prompt_variations = [
                    f"Generate a new multiple-choice question related to {subject}.",
                    f"Create a challenging new question on {subject}.",
                    f"Produce a unique question about {subject}.",
                    f"Design a new MMLU question related to {subject}.",
                ]
                selected_prompt = random.choice(prompt_variations)

                prompt = f"""
                {selected_prompt}
                Here are examples of MMLU questions on the subject of {subject}:

                {examples_str}

                Create a new question on {subject} with four options and indicate the correct answer. Format your response as follows:
                Question: [Your new question]
                A: [Option A]
                B: [Option B]
                C: [Option C]
                D: [Option D]
                Correct answer: [Letter of correct option]
                """

                temp = gen_args.temperature + random.uniform(-0.1, 0.1)

                response = await self.async_client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system",
                         "content": "You are an AI assistant tasked with generating synthetic data for the MMLU dataset."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                    max_tokens=gen_args.max_tokens,
                    top_p=gen_args.top_p,
                    n=batch_size,
                )

                new_questions = [choice.message.content for choice in response.choices]
                filtered_questions = self.filter_similar_questions(new_questions, gen_args.similarity_threshold)
                unique_filtered_questions = [q for q in filtered_questions if q not in self.generated_questions]
                self.generated_questions.update(unique_filtered_questions)

                return unique_filtered_questions

            except Exception as e:
                logging.error(f"Error generating questions for {subject} (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    logging.error(
                        f"Failed to generate questions for {subject} after {self.config.max_retries} attempts")
                    return []

    def filter_similar_questions(self, new_questions: List[str], threshold: float) -> List[str]:
        filtered_questions = []
        for question in new_questions:
            if all(self.calculate_similarity(question, existing_question) < threshold for existing_question in
                   self.generated_questions):
                filtered_questions.append(question)
        return filtered_questions

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1, text2).ratio()

    async def process_subject(self, subject: str, samples: List[dict], gen_args: GenerationArgs):
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        total_questions = 0
        max_questions = gen_args.examples_per_subject
        batch_size = self.config.num_completions

        async def process_batch():
            nonlocal total_questions
            async with semaphore:
                batch = await self.generate_question_batch(subject, samples, gen_args, batch_size)
                total_questions += len(batch)
                return batch

        results = []
        with tqdm(total=max_questions, desc=f"Generating {subject}") as pbar:
            while total_questions < max_questions:
                remaining = max_questions - total_questions
                num_batches = min(self.config.max_parallel_requests, (remaining + batch_size - 1) // batch_size)
                batch_tasks = [process_batch() for _ in range(num_batches)]
                batch_results = await asyncio.gather(*batch_tasks)
                new_questions = [item for sublist in batch_results for item in sublist]
                results.extend(new_questions)
                pbar.update(len(new_questions))

                if not any(batch_results):
                    logging.warning(f"No new questions generated for {subject}. Breaking loop.")
                    break

        return results[:max_questions]


def get_mmlu_samples_by_subject(num_samples_per_subject: int = 10) -> Dict[str, List[dict]]:
    dataset = load_dataset("cais/mmlu", "all")
    samples_by_subject = defaultdict(list)

    for sample in dataset["test"]:
        if len(samples_by_subject[sample["subject"]]) < num_samples_per_subject:
            samples_by_subject[sample["subject"]].append(sample)

    return dict(samples_by_subject)


def parse_generated_question(text: str, subject: str) -> Optional[dict]:
    question_pattern = re.compile(r"Question:\s*(.+)")
    choice_pattern = re.compile(r"([A-D]):\s*(.+)")
    answer_pattern = re.compile(r"Correct answer:\s*([A-D])")

    question_match = question_pattern.search(text)
    if not question_match:
        return None
    question = question_match.group(1).strip()

    choices = {}
    for match in choice_pattern.finditer(text):
        choices[match.group(1)] = match.group(2).strip()

    if len(choices) != 4 or set(choices.keys()) != set("ABCD"):
        return None

    answer_match = answer_pattern.search(text)
    if not answer_match or answer_match.group(1) not in "ABCD":
        return None
    answer = ord(answer_match.group(1)) - ord("A")

    return {
        "question": question,
        "subject": subject,
        "choices": [choices[letter] for letter in "ABCD"],
        "answer": answer,
    }


def upload_to_huggingface(data: List[dict], dataset_name: str):
    dataset = Dataset.from_dict(
        {
            "question": [item["question"] for item in data],
            "subject": [item["subject"] for item in data],
            "choices": [item["choices"] for item in data],
            "answer": [item["answer"] for item in data],
        }
    )

    dataset.push_to_hub(dataset_name)
    print(f"Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{dataset_name}")


async def main():
    config = LLMGenerationConfig()
    processor = LLMProcessor(config)
    gen_args = GenerationArgs()

    samples_by_subject = get_mmlu_samples_by_subject(gen_args.few_shot_examples + gen_args.examples_per_subject)

    all_synthetic_data = []
    for subject, samples in samples_by_subject.items():
        print(f"Starting generation for subject: {subject}")
        raw_synthetic_data = await processor.process_subject(subject, samples, gen_args)
        synthetic_data = [
            parsed_question
            for data in raw_synthetic_data
            if data is not None
            if (parsed_question := parse_generated_question(data, subject)) is not None
        ]
        all_synthetic_data.extend(synthetic_data)
        print(f"Completed generation for subject: {subject}. Generated {len(synthetic_data)} valid questions.")

        # Save progress after each subject
        with open(f"synthetic_mmlu_data_{config.model}_progress.json", "w") as f:
            json.dump(all_synthetic_data, f, indent=2)

    with open(f"synthetic_mmlu_data_{config.model}_final.json", "w") as f:
        json.dump(all_synthetic_data, f, indent=2)

    print(
        f"Generated {len(all_synthetic_data)} valid synthetic MMLU questions across {len(samples_by_subject)} subjects, saved to 'synthetic_mmlu_data_{config.model}_final.json'"
    )

    # Upload to Hugging Face
    upload_to_huggingface(all_synthetic_data, "ai2-adapt-dev/synthetic-mmlu-dataset")


if __name__ == "__main__":
    asyncio.run(main())