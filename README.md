# Linguistic Characterization of Contentious Topics Online

## Case Studies on Contentiousness in Abortion, Climate Change, and Gun Control

Download the complete data, including all features, [here](https://drive.google.com/drive/folders/1xbCLYsCouM1xQ44riU3BjTl-oAtOnRHe?usp=sharing)

Then, download the discourse model [here](https://drive.google.com/file/d/19nBGPPnqaUNN0gGFW1dgYZ_N1EXjZhXB/view?usp=sharing)

To get the BERT representations, use `compute_bert_representations.py`, then use `contentiousness_pipeline.py` to
perform the experiments.

Example: `python contentiousness_pipeline.py --topic abortion --data keyword_data/abortion.pickle`

Requires `python>=3.8`.