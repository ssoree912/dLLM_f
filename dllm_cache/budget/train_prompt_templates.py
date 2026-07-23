from __future__ import annotations

from typing import Final, Protocol


class TrainPromptSample(Protocol):
    dataset: str
    task: str
    context: str
    question: str
    answer_prefix: str


DATASET_PROMPTS: Final[dict[str, str]] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, using a single "
        "phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\n"
        "Answer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely "
        "as you can, using a single phrase or sentence if possible. If the question cannot be "
        'answered based on the information in the article, write "unanswerable". If the question '
        'is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any '
        "explanation.\n\n"
        "Article: {context}\n\n"
        " Answer the question based on the above article as concisely as you can, using a single "
        "phrase or sentence if possible. If the question cannot be answered based on the "
        'information in the article, write "unanswerable". If the question is a yes/no question, '
        'answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
        "Question: {input}\n\n"
        "Answer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n"
        "{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\n"
        "Summary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\n"
        "Answer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\n"
        "Summary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n"
        "{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not "
        "output any other words. The following are some examples.\n\n"
        "{context}\n\n"
        "{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n"
        "{input}"
    ),
}
DATASET_ALIASES: Final[dict[str, str]] = {
    "2wikimultihopqa": "2wikimqa",
    "2wikimultihopqa_train": "2wikimqa",
}
INPUT_PREFIXES: Final[tuple[str, ...]] = ("Question:", "Query:")


def build_train_prompt(sample: TrainPromptSample) -> str:
    dataset = DATASET_ALIASES.get(sample.dataset, sample.dataset)
    template = DATASET_PROMPTS.get(dataset)
    if template is not None:
        return template.format(context=sample.context.strip(), input=prompt_input(sample))
    return build_generic_prompt(sample)


def prompt_input(sample: TrainPromptSample) -> str:
    text = sample.question.strip()
    for prefix in INPUT_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def build_generic_prompt(sample: TrainPromptSample) -> str:
    parts: list[str] = []
    if sample.context.strip():
        parts.append(sample.context.strip())
    parts.append(sample.question.strip())
    if sample.answer_prefix.strip():
        parts.append(sample.answer_prefix.strip())
    elif sample.task in {"Multi-Document QA", "Single-Document QA"}:
        parts.append("Answer:")
    return "\n\n".join(parts)
