import torch
import re
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, PreTrainedTokenizer

class Attention:
    def __init__(self, model_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        if any(model in model_name.lower() for model in ["gpt", "llama", "gemma"]):
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", output_attentions=True, **kwargs)
        else:
            self.model = AutoModel.from_pretrained(model_name, device_map="auto", output_attentions=True, **kwargs)

    def get_attention_for_texts(self, text_x: str, text_y: str=None):
        # text_x = text_x.replace('\n', self.tokenizer.sep_token)
        # if text_y:
        #     text_y = text_y.replace('\n', self.tokenizer.sep_token)
        
        if hasattr(self.model, 'generate'):
            input_ids = self.tokenizer(text_x, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model.generate(input_ids, max_new_tokens=1, output_attentions=True, return_dict_in_generate=True, stopping_criteria=None)
            generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            attentions = outputs.attentions

            return generated_text, attentions, self.tokenizer, input_ids, outputs
        else:
            text_combined = '\n'.join([text_x, text_y])
            input_ids = self.tokenizer(text_combined, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model(input_ids, output_attentions=True)
            attentions_combined = outputs.attentions

            tokens_x = self.tokenizer(text_x, return_tensors="pt").input_ids.to("cuda")
            x_seq_len = tokens_x.size(1)

            return x_seq_len, attentions_combined, self.tokenizer, input_ids, outputs

class ISA:
    def __init__(self, generated_sequences, attentions, tokenizer):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.attention_type = 'enc' if isinstance(attentions[0], torch.Tensor) else 'dec'

        # 문장별로 분리
        # sentences = [sentence.strip() for sentence in generated_text.split('\n') if sentence.strip()]
        sentence_boundaries_text, sentence_boundaries_ids, sentences = self._find_sentence_boundaries(generated_sequences)

        # 문장 간 attention 계산
        layer_attentions = self._integrate_steps(attentions)
        head_attentions = self._integrate_layers(layer_attentions)
        sentence_attention_heads = self._calculate_sentence_attention(head_attentions, sentence_boundaries_ids)
        sentence_attention = self._aggregate_attention(sentence_attention_heads)

        self.sequences = generated_sequences
        self.sentence_boundaries_text = sentence_boundaries_text
        self.sentence_boundaries_ids = sentence_boundaries_ids
        self.sentences = sentences
        self.attentions = attentions
        self.layer_attentions = layer_attentions
        self.head_attentions = head_attentions
        self.sentence_attention_heads = sentence_attention_heads
        self.sentence_attention = sentence_attention
    
    def _integrate_steps(self, attentions):
        if self.attention_type == 'enc':
            # 문장 관계
            # shape: (num_layers, batch_size, num_heads, max_seq_length, max_seq_length)
            return attentions
        else:
            # 생성 모델
            self.attention_type = 'dec'
            num_layers = len(attentions[0])  # 레이어 수
            batch_size, num_heads = attentions[0][0].shape[:2]  # 배치 크기, 헤드 수
            final_seq_len = attentions[-1][-1].shape[-1]  # 최종 시퀀스 길이

            # 최종 합쳐진 어텐션을 저장할 텐서
            integrated = torch.zeros(
                num_layers, batch_size, num_heads, final_seq_len, final_seq_len
            ).to(attentions[0][0].device)

            for step_idx, step_attention in enumerate(attentions):
                for layer_idx, layer_attention in enumerate(step_attention):
                    current_seq_len = layer_attention.shape[-1]  # 현재 스텝의 시퀀스 길이

                    if step_idx == 0:
                        # 첫 스텝은 입력 토큰의 어텐션으로 채움
                        integrated[layer_idx, :, :, :current_seq_len, :current_seq_len] = layer_attention
                    else:
                        # 새로 생성된 토큰의 어텐션 값만 추가
                        integrated[layer_idx, :, :, current_seq_len-1:current_seq_len, :current_seq_len] = layer_attention
        
            # shape: (num_layers, batch_size, num_heads, max_seq_length, max_seq_length)
            return integrated

    def _integrate_layers(self, attentions):
        if self.attention_type == 'enc':
            # 문장 관계
            # attentions shape: (num_layers, batch_size, num_heads, max_seq_length, max_seq_length)
            _, num_heads, max_seq_length, _ = attentions[0].shape

            # 레이어별 어텐션을 합치기 위해 (num_layers, batch_size, num_heads, max_seq_length, max_seq_length)로 스택
            stacked_attentions = torch.stack(attentions, dim=0)  # shape: (num_layers, batch_size, num_heads, max_seq_length, max_seq_length)

            # 어텐션 결합 방법: Max pooling
            integrated = torch.max(stacked_attentions, dim=0)[0]  # (batch_size, num_heads, max_seq_length, max_seq_length)
        else:
            # num_layers, batch_size, num_heads, max_seq_length, _ = attentions.shape[0]

            # Max pooling
            integrated = torch.max(attentions, dim=0)[0]  # shape: (batch_size, num_heads, max_seq_length, max_seq_length)

        # shape: (batch_size, num_heads, max_seq_length, max_seq_length)
        return integrated
    
    def _aggregate_attention(self, sentence_attentions):
        # input shape: (1, num_heads, num_sentences, num_sentences)
        max_attention_heads, _ = torch.max(sentence_attentions, dim=1)

        return max_attention_heads.squeeze(0) # shape: (num_sentences, num_sentences)

    def _remove_tags(self, text):
        # 태그 제거
        return re.sub(r'<[^>]+>', '', text)
    
    def _find_sentence_boundaries(self, sequences):
        newline_token_id = self.tokenizer.convert_tokens_to_ids('\n')
        sentences = []
        text_boundaries = [0]
        sequence_boundaries = [0]
        current_position = 0

        for idx, token in enumerate(sequences):
            token_id = token.to('cpu').item()
            if token_id == newline_token_id:
                current_text = self.tokenizer.decode(sequences[:idx])
                current_position = len(current_text)

                sentences.append(current_text[text_boundaries[-1]:])
                text_boundaries.append(current_position)
                sequence_boundaries.append(idx)
        
        return text_boundaries, sequence_boundaries, sentences

    def _calculate_sentence_attention(self, attentions, sentence_boundaries_ids):
        # 문장 간 attention 계산
        _, num_heads, _, seq_length = attentions.shape # shape: (1, num_heads, seq_length, seq_length)
        num_sentences = len(sentence_boundaries_ids) - 1 # 시작점과 끝점을 포함하므로 1 뺌

        # 0으로 초기화
        sentence_attentions = torch.zeros((1, num_heads, num_sentences, num_sentences))

        # for layer in range(num_layers):
        # layer = 0
        for head in range(num_heads):
            for i in range(num_sentences):
                start_i = sentence_boundaries_ids[i]
                end_i = sentence_boundaries_ids[i + 1]
                for j in range(num_sentences):
                    start_j = sentence_boundaries_ids[j]
                    end_j = sentence_boundaries_ids[j + 1]
                    # 문장 범위 내 텐서 추출
                    attention_slice = attentions[0, head, start_i:end_i, start_j:end_j]
                    
                    if attention_slice.numel() == 0: # 빈 텐서 검사
                        max_attention = 0
                    else:
                        max_attention = torch.max(attention_slice).item() # 문장 내 토큰들의 attention의 최대값
                    
                    sentence_attentions[0, head, i, j] = max_attention # 각 헤드 내에서 i번째 문장관 j번째 문장의 attention

        return sentence_attentions # shape: (1, num_heads, num_sentences, num_sentences)
    
    def get_sentence_token_attention(self, sentence_x_idx=None, sentence_y_idx=None):
        """
        BertViz 시각화에 필요한 데이터를 준비하는 함수
        
        Parameters:
            sentence_x_idx: 첫 번째 문장의 인덱스
            sentence_y_idx: 두 번째 문장의 인덱스
        
        Returns:
            attention_data: bertviz 포맷의 어텐션 데이터
            tokens: 토큰 리스트
            sentence_b_start: 두 번째 문장의 시작 인덱스 (해당하는 경우)
        """
        # 토큰 준비
        tokens = self.tokenizer.convert_ids_to_tokens(self.sequences)

        # 문장 경계 처리
        x_start = self.sentence_boundaries_ids[sentence_x_idx]
        x_end = self.sentence_boundaries_ids[sentence_x_idx + 1]
        y_start = self.sentence_boundaries_ids[sentence_y_idx]
        y_end = self.sentence_boundaries_ids[sentence_y_idx + 1]

        # 두 문장의 토큰 결합
        combined_tokens = tokens[x_start:x_end] + tokens[y_start:y_end]

        # 어텐션 데이터 준비
        if self.attention_type == 'dec':  # 생성 모델의 경우
            num_layers = len(self.attentions[0])
            attention_data = []

            for layer in range(num_layers):
                sent_x_self_attention = self.layer_attentions[layer, :, :, x_start:x_end, x_start:x_end]  # x->x
                sent_y_self_attention = self.layer_attentions[layer, :, :, y_start:y_end, y_start:y_end]  # y->y
                sent_x_to_y_attention = self.layer_attentions[layer, :, :, x_start:x_end, y_start:y_end]  # x->y
                sent_y_to_x_attention = self.layer_attentions[layer, :, :, y_start:y_end, x_start:x_end]  # y->x

                # Attention 결합
                sent_x_combined = torch.cat((sent_x_self_attention, sent_x_to_y_attention), dim=3)
                sent_y_combined = torch.cat((sent_y_to_x_attention, sent_y_self_attention), dim=3)
                combined_attention = torch.cat((sent_x_combined, sent_y_combined), dim=2)

                layer_attention = combined_attention[0].unsqueeze(0)  # batch 차원 추가
                attention_data.append(layer_attention)
        else:  # BERT 계열 모델의 경우
            attention_data = [att.unsqueeze(0) for att in self.attentions]  # batch 차원 추가

        return attention_data, combined_tokens, x_end-x_start
