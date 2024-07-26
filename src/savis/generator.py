import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

class TextGenerator:
    def __init__(self, model_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", output_attentions=True, **kwargs)

    def generate_text(self, input_text, max_new_tokens=100, stop_newline=True):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        if stop_newline:
            stopping_criteria = StoppingCriteriaList([StopOnNewLineCriteria(self.tokenizer)])
        else:
            stopping_criteria = None
        outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, output_attentions=True, return_dict_in_generate=True, stopping_criteria=stopping_criteria)
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        attentions = outputs.attentions
        return generated_text, attentions, self.tokenizer, input_ids, outputs

class StopOnNewLineCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # 마지막으로 생성된 토큰이 줄바꿈인지 확인
        if input_ids[0, -1] == self.tokenizer.eos_token_id or \
            self.tokenizer.decode(input_ids[0, -1]) == "\n":
            return True
        return False
