import os
import json
from tqdm import tqdm
from loader import CruxEval
from llm import Message
from llm.platform_api import OpenAIChat
from evaluate import verify_correctness
from perturbations.structural.renaming import get_local_variables
import multiprocessing

def main():
    model = OpenAIChat(model_name="gpt-5", max_tokens=8096)
    dataset = CruxEval()

    prompt_template = """
You are an expert Python programmer. Given a Python function and its function call, please state all the local variables defined within the function at each line of code, and then predict the output of the function call.

### Python Function:
```python
{code}
```

### Function Call:
```python
{function_call}
```

### Local Variables:
- {local_vars}

State the values of local variables line by line using a markdown table, and then provide the final output of the function call.
Please wrap the table within <table> and </table> tags and the output within <output> and </output> tags.
Remember keep the format correct, for example, use quotes for strings ("<string>").

### Output Format:
<table>
| Line | Code | Variable 1 | Variable 2 | ... | Variable N |
|------|------|------------|------------|-----|------------|
| 1    | ...  | ...        | ...        | ... | ...        |
| 2    | ...  | ...        | ...        | ... | ...        |
| 4 (iteration 1)    | ...  | ...        | ...        | ... | ...        |
| 4 (iteration 2)    | ...  | ...        | ...        | ... | ...        |
| ...  | ...  | ...        | ...        | ... | ...        |
</table>
<output>
{function_call} == <function_output>
</output>
"""

    filename = "gpt5-variable-tracking.jsonl"
    max_retry = 5

    if os.path.exists(filename):
        with open(filename, "r") as f:
            record = [json.loads(line) for line in f.readlines()]
            existing_task_ids = {entry["task_id"] for entry in record}
    else:
        record = []
        existing_task_ids = set()

    def extract(text: str):
        if "<table>" in text and "</table>" in text and "<output>" in text and "</output>" in text:
            table_content = text.split("<table>")[1].split("</table>")[0].strip()
            assertion = text.split("<output>")[1].split("</output>")[0].strip()
            output = assertion.split("==")[-1].strip()
            return table_content, output
        return None, None

    for question in tqdm(dataset.questions_list):
        if question.task_id in existing_task_ids:
            continue
        task_id = question.task_id
        code = question.code
        function_call = question.function_call
        output = question.output
        function_name = question.function_name
        local_vars = get_local_variables(code)
        local_vars_str = ", ".join(local_vars)
        prompt = prompt_template.format(code=code, function_call=function_call, local_vars=local_vars_str).strip()

        for attempt in range(max_retry):
            input_messages = [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content=prompt),
            ]
            responses = model.chat(input_messages)
            table_content, output_content = extract(responses[0])
            if table_content is not None and output_content is not None:
                test = f"{function_call} == {output_content}"
                status = verify_correctness(code=code, test=test)
                if status["status"] == "passed":
                    record.append({
                        "task_id": task_id,
                        "code": code,
                        "function_call": function_call,
                        "function_name": function_name,
                        "expected_output": output,
                        "response": responses[0],
                        "table_content": table_content,
                        "predicted_output": output_content,
                    })
                    with open(filename, "a") as f:
                        f.write(json.dumps(record[-1]) + "\n")
                    break
                else:
                    print(f"[Incorrect] Attempt {attempt + 1} failed for task {task_id}. Retrying...")
            else:
                print(f"[Extraction] Attempt {attempt + 1} failed for task {task_id}. Retrying...")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()