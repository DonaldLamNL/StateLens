import json
import markdown
from io import StringIO
import pandas as pd

with open("gpt5-variable-tracking.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

def md_to_df_via_html(md: str) -> pd.DataFrame:
    html = markdown.markdown(md, extensions=["tables"])
    return pd.read_html(StringIO(html))[0]

PROMPT = """
Given the code snippet:
```python
{code}
```
and the function call with input arguments:
```python
{function_call}
```
Predict the exact output value for `{function_call}`, execute the program step by step before arriving at an answer within the tokens <think> and </think>, and output your prediction using the special tokens <answer> {function_call} == ?? </answer>. Ensure the provided expression syntax is correct!
"""

TEMPLATE = """
<think>
{thought}
</think>
<answer>
{answer}
</answer>
"""

for i, entry in enumerate(data):
    raw_code = entry["code"]
    function_call = entry["function_call"]
    table_content = entry["table_content"]
    predicted_output = entry["predicted_output"]
    answer = f"{function_call} == {predicted_output}"
    df = md_to_df_via_html(table_content)
    
    thoughts = ""
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        line = row_dict.pop("Line")
        code = row_dict.pop("Code").replace("`", "")

        for k, v in row_dict.items():
            if isinstance(v, str) and v.strip("—") == "":
                row_dict[k] = "undefined"
            elif isinstance(v, str) and v.strip("-") == "":
                row_dict[k] = "undefined"

        vars_str = ", ".join(f"{k}={v}" for k, v in row_dict.items())
        
        thought = f"Line {line}: `{code}`; {vars_str}"
        thoughts += thought + "\n"

    prompt = PROMPT.format(code=raw_code, function_call=function_call).strip()
    temp = TEMPLATE.format(thought=thoughts.strip(), answer=answer).strip()
    
    train_script = {
        "system": "You are an expert Python programmer",
        "conversations": [
            {"from": "user", "value": prompt},
            {"from": "assistant", "value": temp}
        ],
        "mask": "user",
        "type": "VALUE_TO_TEXT"
    }

    with open("data_gpt_trace.jsonl", "a") as f:
        json.dump(train_script, f)
        f.write("\n")
