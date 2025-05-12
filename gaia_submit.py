import requests
import yaml
from smolagents import CodeAgent, HfApiModel
from tools.final_answer import FinalAnswerTool

GAIA_API = "https://gaia-api.spaces.huggingface.tech"
HF_USERNAME = "RoomSamurai"
SPACE_CODE_URL = f"https://huggingface.co/spaces/{HF_USERNAME}/AiAgentCourse_final_agent/tree/main"

with open("prompts.yaml", "r") as f:
    prompt_templates = yaml.safe_load(f)

model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", temperature=0.5, max_tokens=2096)
final_answer = FinalAnswerTool()

agent = CodeAgent(
    model=model,
    tools=[final_answer],
    prompt_templates=prompt_templates,
    max_steps=6
)

def get_questions():
    r = requests.get(f"{GAIA_API}/questions")
    return r.json()

def submit_all(answers):
    payload = {
        "username": HF_USERNAME,
        "agent_code": SPACE_CODE_URL,
        "answers": answers
    }
    r = requests.post(f"{GAIA_API}/submit", json=payload)
    return r.json()

if __name__ == "__main__":
    print("Fetching GAIA questions...")
    questions = get_questions()

    submitted = []

    for q in questions[:20]:
        print("\nQuestion:", q["question"])
        try:
            response = agent.run(q["question"])
        except Exception as e:
            print("Agent error:", e)
            response = "error"

        submitted.append({
            "task_id": q["id"],
            "submitted_answer": str(response).strip()
        })

    print("\nSubmitting answers...")
    result = submit_all(submitted)
    print("Submission result:", result)
