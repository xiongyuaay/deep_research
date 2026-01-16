import asyncio
import json

from gpt_researcher import GPTResearcher

SAFE_PROMPT = (
    "You are a safety-aligned assistant. If the user's request involves harm, "
    "refuse briefly and offer a safe, high-level alternative. Do not provide "
    "operational instructions."
)

async def make_response(user_prompt: str) -> str:
    researcher = GPTResearcher(
        query=f"{SAFE_PROMPT}\n\nUser request: {user_prompt}",
        report_type="custom_report",
        report_source="local"  # avoid web retrieval for unsafe prompts
    )
    await researcher.conduct_research()
    return await researcher.write_report()

def main():
    with open("StrongREJECT_train.jsonl", "r", encoding="utf-8") as f, \
        open("strongreject_safe_answers.jsonl", "w", encoding="utf-8") as out:
        for line in f:
            item = json.loads(line)
            response = asyncio.run(make_response(item["prompt"]))
            out.write(json.dumps(
                {"prompt": item["prompt"], "response": response},
                ensure_ascii=False
            ) + "\n")

if __name__ == "__main__":
    main()
