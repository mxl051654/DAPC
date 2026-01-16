from openai import OpenAI, AzureOpenAI
import os, time, json, re


def format_messages(messages, variables={}):
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

    # find all the keys that are still in the prompt using regex [[STR]] where STR is alnum witout space
    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if len(keys_still_in_prompt) > 0:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages


class OpenAI_Model:

    def __init__(self, model, base_url="http://localhost:8022/v1", api_key="dummy"):

        if 'gpt' in model:
            self.API_KEY = ''
            self.BASE_URL = ''
        elif model == 'deepseek-chat':

            self.API_KEY = 'sk-31f42cabd232499bbb43b84d34d1687c'
            self.BASE_URL = 'https://api.deepseek.com'
            # self.BASE_URL = 'https://api.deepseek.com/v1'
        else:
            self.API_KEY = api_key
            self.BASE_URL = base_url

        self.client = OpenAI(api_key=self.API_KEY, base_url=self.BASE_URL)
        self.model = model

    def cost_calculator(self, usage, is_batch_model=False):
        is_finetuned, base_model = False, self.model

        if self.model.startswith("ft:gpt"):
            is_finetuned = True
            base_model = model.split(":")[1]

        prompt_tokens = usage['prompt_tokens']
        if 'prompt_tokens_details' in usage:
            prompt_tokens_cached = usage['prompt_tokens_details']['cached_tokens']
        else:
            prompt_tokens_cached = 0
        prompt_tokens_non_cached = prompt_tokens - prompt_tokens_cached

        completion_tokens = usage['completion_tokens']
        if base_model.startswith("gpt-4o-mini"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.0003, 0.00015
            else:
                inp_token_cost, out_token_cost = 0.00015, 0.0006
        elif base_model.startswith("gpt-4o"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.00375, 0.015
            else:
                inp_token_cost, out_token_cost = 0.0025, 0.01
        elif base_model.startswith("gpt-3.5-turbo"):
            inp_token_cost, out_token_cost = 0.0005, 0.0015
        elif base_model.startswith("o1-mini"):
            inp_token_cost, out_token_cost = 0.003, 0.012
        elif base_model.startswith("gpt-4.5-preview"):
            inp_token_cost, out_token_cost = 0.075, 0.150
        elif base_model.startswith("o1-preview") or base_model == "o1":
            inp_token_cost, out_token_cost = 0.015, 0.06

        # NOTE
        elif base_model.startswith("deepseek"):
            inp_token_cost, out_token_cost = 1, 2
        else:
            raise Exception(f"Model {model} pricing unknown, please add")

        cache_discount = 0.5  # cached tokens are half the price
        batch_discount = 0.5  # batch API is half the price
        total_usd = ((prompt_tokens_non_cached + prompt_tokens_cached * cache_discount) / 1000) * inp_token_cost + (
                completion_tokens / 1000) * out_token_cost
        if is_batch_model:
            total_usd *= batch_discount

        return total_usd

    def generate(
            self, messages, timeout=60, max_retries=3, temperature=1.0, is_json=False,
            return_metadata=False, max_tokens=512, variables={},
    ):

        kwargs = {}
        if is_json:
            kwargs["response_format"] = {"type": "json_object"}  # NOTE
        N = 0

        # messages = format_messages(messages, variables)
        if (self.model.startswith("o1") and len(messages) > 1
                and messages[0]["role"] == "system"
                and messages[1]["role"] == "user"):
            system_message = messages[0]["content"]
            messages[1]["content"] = f"System Message: {system_message}\n{messages[1]['content']}"
            messages = messages[1:]

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, timeout=timeout,
                    # max_completion_tokens=max_tokens,
                    max_tokens=max_tokens,  # NOTE openai 版本接口变更
                    temperature=temperature,
                    stream=False,
                    **kwargs
                )
                break
            except Exception as e:
                print('Exception :', e)
                N += 1
                if N >= max_retries:
                    raise Exception("Failed to get response from OpenAI")
                else:
                    time.sleep(2)
                    timeout += 6
                    print('retry')

        response = response.to_dict()
        usage = response['usage']
        response_text = response["choices"][0]["message"]["content"]
        # total_usd = self.cost_calculator(model, usage)
        prompt_tokens_cached = 0
        if usage.get('prompt_tokens_details'):
            prompt_tokens_cached = usage['prompt_tokens_details']['cached_tokens']

        if not return_metadata:
            return response_text
        return {
            "message": response_text,
            "total_tokens": usage['total_tokens'],
            "prompt_tokens": usage['prompt_tokens'],
            "prompt_tokens_cached": prompt_tokens_cached,
            "completion_tokens": usage['completion_tokens'],
            "total_usd": usage['total_tokens'] + usage['prompt_tokens'],
        }

    def generate_json(self, messages, **kwargs):
        response = self.generate(messages, is_json=True, **kwargs)
        response["message"] = json.loads(response["message"])
        return response


def meta(model='deepseek-chat', messages=None):

    API_KEY = 'sk-31f42cabd232499bbb43b84d34d1687c'
    BASE_URL = 'https://api.deepseek.com'
    # BASE_URL = 'https://api.deepseek.com/v1'

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    response = client.chat.completions.create(
        model=model, messages=messages, timeout=120,
        max_completion_tokens=4096,
        temperature=1.0,
    )
    return response


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Humor is a way to make people laugh."
                                    " Tell me a joke about UC Berkeley. response in json format."},
        # {"role": "assistant", "content": '{"joke": '}
    ]

    messages = [
        {
            'role': 'user',
            'content': '\n        Here\'s a series of shards, help me design three variations of each shard. \n        You don\'t need to keep the core logic and semantics consistent.\n        Make sure that the answer to the question has a substantial impact. \n        Return in JSON format.\n\n        Response Demo:\n        {\n            "shards": [\n                {\n                  "shard_id": 1,\n                  "original": "Turn digits into names in a list",\n                  "variants": [\n                    "Turn digits into names, but map 1-5 to "One" to "Five", and 6-9 to "Six" to "Nine"",\n                    "Turn digits into names and join them into a single string",\n                    "Turn digits into names, but only for even numbers"\n                  ]\n                }\n                {\n                  "shard_id": 2,\n                  ...\n            ]\n        }\n\n        SHARDS: \n        [\n    {\n        "shard_id": 1,\n        "shard": "how much money is left after buying two chocolates?"\n    },\n    {\n        "shard_id": 2,\n        "shard": "I need to buy exactly two chocolates and can\'t end up with negative money"\n    },\n    {\n        "shard_id": 3,\n        "shard": "try to minimize the cost of those two chocolates"\n    },\n    {\n        "shard_id": 4,\n        "shard": "if I can\'t buy two chocolates without debt, just return the initial money I have"\n    },\n    {\n        "shard_id": 5,\n        "shard": "I\'ve got an array of chocolate prices to work with"\n    },\n    {\n        "shard_id": 6,\n        "shard": "there\'s also an initial amount of cash I start with"\n    },\n    {\n        "shard_id": 7,\n        "shard": "example: if prices are [1, 2, 2] and I\'ve got 3 cash, I end up with 0 leftover"\n    },\n    {\n        "shard_id": 8,\n        "shard": "the number of chocolate prices is between 2 and 50"\n    },\n    {\n        "shard_id": 9,\n        "shard": "each chocolate price is between 1 and 100"\n    },\n    {\n        "shard_id": 10,\n        "shard": "the initial amount of money is between 1 and 100"\n    }\n]\n        \n        Design three variations of each shard, return in JSON format.\n        '
        }
    ]

    llm = OpenAI_Model(model='deepseek-chat')
    # llm = OpenAI_Model(model='/data/hf/Qwen/Qwen2.5-3B-Instruct')

    # response = llm.generate(messages, return_metadata=True)
    response = llm.generate_json(
        messages,
        temperature=0.5, return_metadata=True, max_tokens=4096)['message']

    # response = meta(model='deepseek-chat', messages=messages)

    print(response)
