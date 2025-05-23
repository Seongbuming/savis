# SAVIS: Sentence-Level Attention Visualization for Large Language Models

**SAVIS** (**S**entence-Level **A**ttention **Vis**ualization) is a Python library for visualizing attention across sentences in texts processed by large language models (LLMs). This tool enhances the interpretability of LLMs by providing an intuitive visualization of how attention is distributed across sentences in generated text.

This work has been accepted to NAACL 2025 (Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics) Demo. [📄 Read the paper](https://aclanthology.org/2025.naacl-demo.27). Citation details are available in the [Citation](#citation) section.

<center>
<img src="https://raw.githubusercontent.com/Seongbuming/savis/master/images/savis.png" width="500" alt="SAVIS analyzing reviews">
</center>

## Features

- Inter-sentence attention calculation
- Interactive visualization of attention patterns
- Support for various LLMs through [Hugging Face](https://huggingface.co)'s `transformers` library

## Installation

```bash
pip install savis
```

## Quick Start

```python
from savis import TextGenerator, ISA, ISAVisualization

# Initialize the text generator with your chosen model
generator = TextGenerator("Model name")

# Generate text and get attention data
input_text = "Your input prompt here"
generated_text, attentions, tokenizer, input_ids, outputs = generator.generate_text(input_text)

# Calculate inter-sentence attention
isa = ISA(outputs.sequences[0], attentions, tokenizer)

# Visualize the attention patterns
vis = ISAVisualization(isa.sentence_attention, isa.sentences)
vis.visualize_sentence_attention()
```

## Key Components

1. `TextGenerator`: Interfaces with the LLM to generate text and extract attention information.
2. `Attention`: Manages the underlying LLM and provides methods for obtaining attention data from the model.
3. `ISA` (Inter-Sentence Attention): Processes raw attention data to compute attention between sentences.
4. `ISAVisualization`: Creates interactive visualizations of the computed inter-sentence attention.

These components work together to provide a comprehensive pipeline from text generation to attention visualization:

- `TextGenerator` uses the LLM to generate text based on input prompts.
- `Attention` handles the interaction with the LLM, extracting detailed attention information.
- `ISA` takes the raw attention data and computes meaningful inter-sentence attention scores.
- `ISAVisualization` takes these scores and creates interactive visualizations.

## Citation

```bibtex
@inproceedings{seo-etal-2025-sentence,
    title = "A Sentence-Level Visualization of Attention in Large Language Models",
    author = "Seo, Seongbum  and
      Yoo, Sangbong  and
      Lee, Hyelim  and
      Jang, Yun  and
      Park, Ji Hwan  and
      Kim, Jeong-Nam",
    editor = "Dziri, Nouha  and
      Ren, Sean (Xiang)  and
      Diao, Shizhe",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (System Demonstrations)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-demo.27/",
    doi = "10.18653/v1/2025.naacl-demo.27",
    pages = "313--320",
    ISBN = "979-8-89176-191-9"
}
```

## License

SAVIS is released under the MIT License. See the [LICENSE](https://github.com/Seongbuming/savis/blob/master/LICENSE) file for more details.
