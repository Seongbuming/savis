import torch
import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

class Attention:
    def __init__(self, model_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        if any(model in model_name.lower() for model in ["gpt", "llama", "gemma"]):
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", output_attentions=True, **kwargs)
        else:
            self.model = AutoModel.from_pretrained(model_name, device_map="auto", output_attentions=True, **kwargs)

    def get_attention_for_texts(self, text_x, text_y=None):
        if hasattr(self.model, 'generate'):
            input_ids = self.tokenizer(text_x, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model.generate(input_ids, max_new_tokens=1, output_attentions=True, return_dict_in_generate=True, stopping_criteria=None)
            generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            attentions = outputs.attentions

            return generated_text, attentions, self.tokenizer, input_ids, outputs
        else:
            text_combined = text_x + "\n" + text_y
            input_ids = self.tokenizer(text_combined, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model(input_ids, output_attentions=True)
            attentions_combined = outputs.attentions

            tokens_x = self.tokenizer(text_x, return_tensors="pt").input_ids.to("cuda")
            x_seq_len = tokens_x.size(1)

            return x_seq_len, attentions_combined, self.tokenizer, input_ids, outputs

class ISA:
    def __init__(self, generated_sequences, attentions, tokenizer):
        self.tokenizer = tokenizer

        nltk.download('punkt')

        # 문장을 마침표 기준으로 분리
        # sentences = [sentence.strip() for sentence in generated_text.split('\n') if sentence.strip()]
        sentence_boundaries, sentences = self._find_sentence_boundaries(generated_sequences)

        # 문장 간 attention 계산
        integrated_attentions = self._integrate_attentions(attentions)
        sentence_attention_heads = self._calculate_sentence_attention(integrated_attentions, sentence_boundaries)
        sentence_attention = self._aggregate_attention(sentence_attention_heads)

        self.sentences = sentences
        self.sentence_attention_heads = sentence_attention_heads
        self.sentence_attention = sentence_attention
    
    def _integrate_attentions(self, attentions):
        if isinstance(attentions[0], torch.Tensor):
            # 문장 관계
            num_layers = len(attentions)
            _, num_heads, max_seq_length, _ = attentions[0].shape

            integrated = torch.zeros((1, num_heads, max_seq_length-1, max_seq_length-1), device=attentions[0].device)
            stacked_attentions = torch.stack([att[0] for att in attentions], dim=0)
            integrated = torch.max(stacked_attentions[:, :, 1:, 1:], dim=0)[0]
            integrated = integrated.unsqueeze(0) # shape: (1, num_heads, max_seq_length-1, max_seq_length-1)
        else:
            # 생성 모델
            num_layers = len(attentions)
            _, num_heads, prompt_seq_length, _ = attentions[0][-1].shape
            max_seq_length = attentions[-1][-1].shape[-1]

            integrated = torch.zeros((1, num_heads, max_seq_length, max_seq_length))
            integrated[0, :, :prompt_seq_length, :prompt_seq_length] = attentions[0][-1] # 0번째 레이어(프롬프트 간의 attention을 가짐)의 마지막 히든 레이어의 heads들은 그대로 붙임
            # 나머지 레이어들(생성된 토큰들과 이전 토큰들의 attention을 가짐)의 마지막 히든 레이어의 heads들을 이어붙임
            for layer in range(1, num_layers):
                integrated[0, :, prompt_seq_length+layer:prompt_seq_length+layer+1, :prompt_seq_length+layer] = attentions[layer][-1][0]
            
        return integrated # shape: (1, num_heads, max_seq_length, max_seq_length)
    
    def _aggregate_attention(self, sentence_attentions):
        # input shape: (1, num_heads, num_sentences, num_sentences)
        max_attention_heads, _ = torch.max(sentence_attentions, dim=1)

        return max_attention_heads.squeeze(0) # shape: (num_sentences, num_sentences)

    def _remove_tags(self, text):
        # 태그 제거
        return re.sub(r'<[^>]+>', '', text)

    def _find_sentence_boundaries(self, sequences):
        # 전체 텍스트 디코딩
        full_text = self.tokenizer.decode(sequences)

        # 태그 제거
        full_text = self._remove_tags(full_text)

        # 줄바꿈을 기준으로 텍스트 나누기
        lines = full_text.split('\n')

        sentences = []
        boundaries = [0]
        current_position = 0

        for line in lines:
            if line.strip(): # 빈 줄 무시
                # 각 줄에 대해 NLTK sent_tokenize 적용
                line_sentences = sent_tokenize(line)
                sentences.extend(line_sentences)

                for sentence in line_sentences:
                    current_position += len(self.tokenizer.encode(sentence, add_special_tokens=False))
                    boundaries.append(current_position)

        # 마지막 빈 문장 제거
        if sentences and not sentences[-1].strip():
            sentences.pop()
            boundaries.pop()
        
        return boundaries, sentences

    # def _find_sentence_boundaries(self, sequences):
    #     boundaries = [0]
    #     sentences = []

    #     for i, token in enumerate(sequences):
    #         decoded = self.tokenizer.decode(token)
    #         if '\n' in decoded:
    #             sentences.append(self.tokenizer.decode(sequences[boundaries[-1]:i]))
    #             boundaries.append(i)

    #     sentences.append(self.tokenizer.decode(sequences[boundaries[-1]:len(sequences)]))
    #     boundaries.append(len(sequences))

    #     return boundaries, sentences

    def _calculate_sentence_attention(self, attentions, sentence_boundaries):
        # 문장 간 attention 계산
        # num_layers = len(attentions)
        # num_layers = 1
        _, num_heads, _, seq_length = attentions.shape # shape: (1, num_heads, seq_length, seq_length)
        num_sentences = len(sentence_boundaries) - 1 # 시작점과 끝점을 포함하므로 1 뺌

        # 0으로 초기화
        sentence_attentions = torch.zeros((1, num_heads, num_sentences, num_sentences))

        # for layer in range(num_layers):
        # layer = 0
        for head in range(num_heads):
            for i in range(num_sentences):
                start_i = sentence_boundaries[i]
                end_i = sentence_boundaries[i + 1]
                for j in range(num_sentences):
                    start_j = sentence_boundaries[j]
                    end_j = sentence_boundaries[j + 1]
                    # 문장 범위 내 텐서 추출
                    attention_slice = attentions[0, head, start_i:end_i, start_j:end_j]
                    
                    if attention_slice.numel() == 0: # 빈 텐서 검사
                        max_attention = 0
                    else:
                        max_attention = torch.max(attention_slice).item() # 문장 내 토큰들의 attention의 최대값
                    
                    sentence_attentions[0, head, i, j] = max_attention # 각 헤드 내에서 i번째 문장관 j번째 문장의 attention

        return sentence_attentions # shape: (1, num_heads, num_sentences, num_sentences)
