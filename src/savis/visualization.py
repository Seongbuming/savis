import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import mplcursors
import numpy as np
import textwrap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox
from matplotlib.widgets import Slider

class ISAVisualization:
    def __init__(self, sentence_attention, sentences):
        self.sentence_attention = sentence_attention
        self.sentences = sentences

    def visualize_sentence_attention(self, figsize=(15,10)):
        num_sentences = len(self.sentences)

        # fig = plt.figure(figsize=(num_sentences/2+2, num_sentences/3+1))
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, height_ratios=[5, 0.2], width_ratios=[2, 1], hspace=0.3, wspace=0.3)

        ax = plt.subplot(gs[0, 0])
        ax.set_xlim(-0.5, num_sentences-0.5)
        ax.set_ylim(-0.5, num_sentences-0.5)

        ax.set_aspect('equal', adjustable='box')

        x, y = np.meshgrid(np.arange(num_sentences), np.arange(num_sentences))
        x = x.flatten()
        y = y.flatten()
        colors = self.sentence_attention.transpose(0, 1).flatten()

        # 커스텀 그라디언트 색상 맵 생성
        custom_colors = ["black", "blue", "cyan", "lime", "yellow", "orange", "red"]
        cmap = LinearSegmentedColormap.from_list("custom_gradient", custom_colors)

        # colors 배열을 넘파이 배열로 변환
        colors_np = np.array(colors)

        # 컬러바 추가
        ax_cbar = plt.subplot(gs[1, 0])
        norm = plt.Normalize(vmin=np.min(colors_np), vmax=np.max(colors_np))
        cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cbar, orientation='horizontal')
        cb1.ax.tick_params(labelleft=False, labelright=True)
        cb1.set_label('Attention Score')

        # 0인 값은 white로, 그 외는 커스텀 그라디언트를 적용
        color_values = np.array([cmap(val) if val > 0 else (1, 1, 1, 1) for val in colors_np])

        # 모든 포인트를 포함하여 scatter 플롯 생성
        ax_width = figsize[0] / 2
        size = ax_width / num_sentences * 500
        scatter = ax.scatter(x, y, color=color_values, cmap=cmap, s=size, edgecolors='none')

        ax.set_xticks(np.arange(num_sentences))
        ax.set_yticks(np.arange(num_sentences))
        fontsize = 8
        ax.set_xticklabels(labels=np.arange(num_sentences), fontsize=fontsize)
        ax.set_yticklabels(labels=np.arange(num_sentences), fontsize=fontsize)

        # plt.grid(False)

        # 텍스트 영역 추가
        text_ax = plt.subplot(gs[0, 1])
        text_ax.axis('off')  # 축 비활성화

        text = text_ax.text(0, 0.5, "Hover over points", ha='left', va='center', transform=text_ax.transAxes)
        
        def on_motion(event):
            cont, ind = scatter.contains(event)
            if cont:
                index = ind["ind"][0]
                index_x = index % num_sentences
                index_y = index // num_sentences
                generated_sentence = self._wrap_text(self.sentences[index_x], 50)
                focused_sentence = self._wrap_text(self.sentences[index_y], 50)
                attention = f"{self.sentence_attention[index_x, index_y]:.5f}"
                # text.set_text(f"[x = {index_x}] Generated Sentence\n{generated_sentence}\n\n[y = {index_y}] Focused Sentence\n{focused_sentence}\n\nISA: {attention}")
                # text.set_text(f"Generated Sentence: {sentences[index_y]}\n\nFocused Sentence: {sentences[index_x]}\n\nAttention: {sentence_attention[index_y, index_x]}")
                text_ax.clear()
                text_ax.axis('off')
                lines = [
                    (f"[x = {index_x}] Generated Sentence", 'bold'),
                    (generated_sentence, None),
                    (None, None),
                    (f"[y = {index_y}] Focused Sentence", 'bold'),
                    (focused_sentence, None),
                    (None, None),
                    (f"ISA: {attention}", 'bold')
                ]
                packed_lines = VPacker(children=[
                    TextArea(line, textprops=dict(fontweight=weight)) for line, weight in lines
                ], align='left', pad=0, sep=5)
                anchored_box = AnnotationBbox(packed_lines, (0.5, 0.5), frameon=False, xycoords='axes fraction')
                text_ax.add_artist(anchored_box)
            else:
                text.set_text("Hover over points")
            fig.canvas.draw_idle()

        # 마우스 이동 이벤트에 대한 핸들러 등록
        fig.canvas.mpl_connect('motion_notify_event', on_motion)

        plt.show()

    def visualize_sentence_attention_heatmap(self):
        # 문장 간 어텐션 매트릭스의 크기
        num_sentences = self.sentence_attention.shape[0]
        
        # 히트맵 생성
        fig, ax = plt.subplots(figsize=(num_sentences, num_sentences))
        cax = ax.matshow(self.sentence_attention, cmap='viridis')
        
        # 축에 문장을 레이블로 추가
        ax.set_xticks(np.arange(num_sentences))
        ax.set_yticks(np.arange(num_sentences))
        ax.set_xticklabels(self.sentences, rotation=90)
        ax.set_yticklabels(self.sentences)
        
        # 컬러바 추가
        fig.colorbar(cax)
        
        plt.show()

    def visualize_token_attention_heatmap(self, attentions, tokenizer, input_ids, layer=-1, head=None, figsize=(50,50)):
        import seaborn as sns

        # 마지막 레이어의 어텐션 가져오기
        if isinstance(attentions[0], tuple):
            layer_attention = attentions[layer][0]  # shape: (batch_size, num_heads, seq_len, seq_len)
        else:
            layer_attention = attentions[layer]  # shape: (batch_size, num_heads, seq_len, seq_len)

        # 배치 차원과 헤드 차원에 대해 평균 계산
        if head is None:
            attention = layer_attention.mean(dim=(0,1)).cpu().numpy()
        else:
            attention = layer_attention[:, head].mean(dim=0).cpu().numpy()

        # 토큰 디코딩
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        num_tokens = len(tokens)
        print(f"Total number of tokens: {num_tokens}")

        # 커스텀 그라디언트 색상 맵 생성
        custom_colors = ["black", "blue", "cyan", "lime", "yellow", "orange", "red"]
        cmap = LinearSegmentedColormap.from_list("custom_gradient", custom_colors)

        # 히트맵 생성
        plt.figure(figsize=figsize)
        
        # vmin을 0으로, vmax를 1로 설정하여 전체 범위 사용
        vmin, vmax = 0, 1
        
        sns.heatmap(attention, 
                    xticklabels=tokens, 
                    yticklabels=tokens, 
                    cmap=cmap, 
                    square=True,
                    vmin=vmin,
                    vmax=vmax,
                    annot=False,
                    cbar_kws={"shrink": .8})

        plt.title("Token-Level Attention Heatmap", fontsize=20)
        plt.xlabel("Target Tokens", fontsize=16)
        plt.ylabel("Source Tokens", fontsize=16)

        # x축 레이블 회전
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.show()

    def _wrap_text(self, text, width):
        """
        텍스트가 width 사이즈 넘어갈 시 줄바꿈
        """
        return '\n'.join(textwrap.wrap(text, width))

    def visualize_sentence_token_attention_heatmap(self,
        attentions, tokenizer, input_ids, sentence_boundaries, sentences, sent_x_idx, sent_y_idx, layer_idx=None, head_idx=None,
        figsize=(15,10)):
        """
        두 문장 간의 토큰 단위 어텐션 히트맵 시각화
        
        Parameters:
            attentions: 모델의 어텐션 값
            tokenizer: 토크나이저
            input_ids: 입력 토큰 ID
            sentence_boundaries: 문장 경계 인덱스 리스트
            sentences: 문장 리스트
            sent_x_idx: X 문장 인덱스
            sent_y_idx: Y 문장 인덱스
            layer_idx: 시각화할 레이어 인덱스 (None이면 사용자가 선택)
            head_idx: 시각화할 헤드 인덱스 (None이면 사용자가 선택)
        """
        if isinstance(attentions[0], tuple):
            num_layers = len(attentions)
            num_heads = attentions[0][0].shape[1]
        else:
            num_layers = len(attentions)
            num_heads = attentions[0].shape[1]

        # 커스텀 컬러맵 생성
        custom_colors = ["black", "blue", "cyan", "lime", "yellow", "orange", "red"]
        cmap = LinearSegmentedColormap.from_list("custom_gradient", custom_colors)

        # 문장 범위 설정
        x_start = sentence_boundaries[sent_x_idx]
        x_end = sentence_boundaries[sent_x_idx + 1]
        y_start = sentence_boundaries[sent_y_idx]
        y_end = sentence_boundaries[sent_y_idx + 1]

        # 토큰 리스트 생성
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        x_tokens = tokens[x_start:x_end]
        y_tokens = tokens[y_start:y_end]

        # UI 설정
        if layer_idx is None or head_idx is None:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(3, 2, height_ratios=[1, 5, 1], width_ratios=[4, 0.2])
            
            # 문장 텍스트 표시 영역
            ax_sentences = plt.subplot(gs[0, 0])
            ax_sentences.axis('off')
            ax_sentences.text(0.1, 0.7, f"Sentence X ({sent_x_idx}): {sentences[sent_x_idx]}", wrap=True)
            ax_sentences.text(0.1, 0.3, f"Sentence Y ({sent_y_idx}): {sentences[sent_y_idx]}", wrap=True)
            
            # 히트맵 영역과 colorbar 영역
            ax_heatmap = plt.subplot(gs[1, 0])
            ax_colorbar = plt.subplot(gs[1, 1])

            # 슬라이더를 위한 영역
            ax_slider_area = plt.subplot(gs[2, 0])
            ax_slider_area.axis('off')  # 슬라이더 영역의 axes를 보이지 않게 설정
            
            current_layer = 0
            current_head = 0
            current_im = None
            colorbar = None

            def create_sliders():
                # 슬라이더 영역의 위치 가져오기
                slider_bbox = ax_slider_area.get_position()
                
                # 슬라이더 위치 계산
                slider_height = 0.03
                slider_bottom = slider_bbox.y0 + (slider_bbox.height - slider_height) / 2 - 0.05
                
                # 기존 슬라이더 제거
                if hasattr(create_sliders, 'slider_axes'):
                    for ax in create_sliders.slider_axes:
                        ax.remove()
                
                # 새 슬라이더 생성
                slider_ax_layer = plt.axes([slider_bbox.x0, slider_bottom, 0.3, slider_height])
                slider_ax_head = plt.axes([slider_bbox.x0 + 0.4, slider_bottom, 0.3, slider_height])
                
                create_sliders.slider_axes = [slider_ax_layer, slider_ax_head]
                
                slider_layer = Slider(slider_ax_layer, 'Layer', 0, num_layers-1, valinit=current_layer, valstep=1)
                slider_head = Slider(slider_ax_head, 'Head', 0, num_heads-1, valinit=current_head, valstep=1)
                
                return slider_layer, slider_head

            def on_resize(event):
                # 리사이즈 시 슬라이더 재배치
                for ax in [slider_ax_layer, slider_ax_head]:
                    ax.remove()
                create_sliders()
                fig.canvas.draw_idle()

            # 첫 번째 플롯 후 슬라이더 생성을 위해 draw 호출
            fig.canvas.draw()
            slider_layer, slider_head = create_sliders()
            slider_ax_layer = slider_layer.ax
            slider_ax_head = slider_head.ax

            def update_plot(val):
                nonlocal current_layer, current_head, current_im
                current_layer = int(slider_layer.val)
                current_head = int(slider_head.val)
                ax_heatmap.clear()
                plot_attention_heatmap(current_layer, current_head)
                fig.canvas.draw_idle()

            slider_layer.on_changed(update_plot)
            slider_head.on_changed(update_plot)
            fig.canvas.mpl_connect('resize_event', on_resize)
        else:
            # layer_idx와 head_idx가 지정된 경우의 코드는 이전과 동일
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 5], width_ratios=[4, 0.2])
            
            ax_sentences = plt.subplot(gs[0, 0])
            ax_sentences.axis('off')
            ax_sentences.text(0.1, 0.7, f"Sentence X ({sent_x_idx}): {sentences[sent_x_idx]}", wrap=True)
            ax_sentences.text(0.1, 0.3, f"Sentence Y ({sent_y_idx}): {sentences[sent_y_idx]}", wrap=True)
            
            ax_heatmap = plt.subplot(gs[1, 0])
            ax_colorbar = plt.subplot(gs[1, 1])
            current_layer = layer_idx
            current_head = head_idx
            current_im = None
            colorbar = None

        def plot_attention_heatmap(layer, head):
            nonlocal current_im, colorbar
            
            # 어텐션 값 추출
            if isinstance(attentions[0], tuple):
                attention = attentions[layer][0][0, head].cpu().numpy()
            else:
                attention = attentions[layer][0, head].cpu().numpy()

            # 두 문장 간의 어텐션만 추출
            attention_subset = attention[y_start:y_end, x_start:x_end]
            
            # 이전 imshow 객체가 있으면 제거
            if current_im is not None:
                current_im.remove()
            
            # 히트맵 생성 (vmin과 vmax를 0과 1로 고정)
            current_im = ax_heatmap.imshow(attention_subset, cmap=cmap, vmin=0.0, vmax=1.0)
            
            # 토큰 레이블 추가
            ax_heatmap.set_xticks(np.arange(len(x_tokens)))
            ax_heatmap.set_yticks(np.arange(len(y_tokens)))
            ax_heatmap.set_xticklabels(x_tokens, rotation=45, ha='right')
            ax_heatmap.set_yticklabels(y_tokens)
            
            # colorbar가 없을 때만 생성 (수직 방향으로)
            if colorbar is None:
                colorbar = plt.colorbar(current_im, cax=ax_colorbar, orientation='vertical')
                colorbar.set_label('Attention Score')
            else:
                colorbar.update_normal(current_im)
                
            ax_heatmap.set_title(f'Layer {layer+1}, Head {head+1} - Attention between sentences {sent_x_idx} and {sent_y_idx}')

            # 마우스 호버 이벤트 처리
            def on_hover(event):
                if event.inaxes == ax_heatmap:
                    x, y = int(event.xdata), int(event.ydata)
                    if 0 <= x < len(x_tokens) and 0 <= y < len(y_tokens):
                        hover_text = f"From: {y_tokens[y]}\nTo: {x_tokens[x]}\nAttention: {attention_subset[y, x]:.4f}"
                        ax_heatmap.set_title(hover_text)
                        fig.canvas.draw_idle()

            fig.canvas.mpl_connect('motion_notify_event', on_hover)

        # 초기 플롯 생성
        plot_attention_heatmap(current_layer, current_head)
        plt.tight_layout()
        plt.show()