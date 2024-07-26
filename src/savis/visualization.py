import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import mplcursors
import numpy as np
import textwrap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox

class ISAVisualization:
    def __init__(self, sentence_attention, sentences):
        self.sentence_attention = sentence_attention
        self.sentences = sentences

    def visualize_sentence_attention(self):
        num_sentences = len(self.sentences)

        # fig = plt.figure(figsize=(num_sentences/2+2, num_sentences/3+1))
        fig = plt.figure(figsize=(num_sentences/2, num_sentences/3))
        gs = gridspec.GridSpec(2, 2, height_ratios=[5, 0.2], width_ratios=[1, 1], hspace=0.3)

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
        scatter = ax.scatter(x, y, color=color_values, cmap=cmap, s=150, edgecolors='none')

        ax.set_xticks(np.arange(num_sentences))
        ax.set_yticks(np.arange(num_sentences))

        plt.grid(False)

        # 텍스트 영역 추가
        text_ax = plt.subplot(gs[0, 1])
        text_ax.axis('off')  # 축 비활성화

        text = text_ax.text(0, 0.5, "Hover over points", ha='left', va='center', transform=text_ax.transAxes)
        
        def on_motion(event):
            cont, ind = scatter.contains(event)
            if cont:
                index = ind["ind"][0]
                index_x = index // num_sentences
                index_y = index % num_sentences
                generated_sentence = self._wrap_text(self.sentences[index_y], 50)
                focused_sentence = self._wrap_text(self.sentences[index_x], 50)
                attention = f"{self.sentence_attention[index_y, index_x]:.5f}"
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

    def _wrap_text(self, text, width):
        """
        텍스트가 width 사이즈 넘어갈 시 줄바꿈
        """
        return '\n'.join(textwrap.wrap(text, width))

