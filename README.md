# [NeurIPS 2024 Oral] MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making

<p align="center">
   📖 <a href="https://arxiv.org/abs/2404.15155" target="_blank">Paper</a>&nbsp;&nbsp;&nbsp;&nbsp;🤖 <a href="https://mdagents2024.github.io/" target="_blank">Project Page</a>
</p>



Foundation models are becoming invaluable tools in medicine. Despite their promise, the strategic deployment of Large Language Models (LLMs) for effective utility in complex medical tasks remains an open question. We introduce a novel framework, <ins><b>M</b></ins>edical <ins><b>D</b></ins>ecision-making <ins><b>Agents</b></ins> (**MDAgents**) which aims to address this gap by automatically assigning a collaboration structure for a team of LLMs. The assigned solo or group collaboration structure is tailored to the medical task at hand, a simple emulation of how real-world medical decision-making processes adapt to tasks of different complexities. We evaluate our framework and baseline methods with state-of-the-art LLMs across a suite of medical benchmarks containing real-world medical knowledge and challenging clinical diagnosis. MDAgents achieved the best performance in **seven out of ten** benchmarks on the tasks that require an understanding of medical knowledge and multi-modal reasoning, showing a significant improvement of up to **11.8\%** compared to previous multi-agent setting (p < 0.05). Ablation studies reveal that our MDAgents effectively determines medical complexity to optimize for *efficiency* and *accuracy* across diverse medical tasks. We also explore the dynamics of group consensus, offering insights into how collaborative agents could behave in complex clinical team dynamics.

<p align="center">
   <img width="800" alt="image" src="imgs/animation.gif">
   <img width="800" alt="image" src="imgs/case_study.png">
</p>

<br>
<br>

## Quick Start

Create a new virtual environment, e.g. with conda

```bash
~$ conda create -n mdagents python>=3.9
```

Install the required packages:
```bash
~$ pip install -r requirements.txt
```

Activate the environment:
```bash
~$ conda activate mdagents
```

Set up API keys:
```bash
~$ export openai_api_key="your_openai_api_key_here"
~$ export genai_api_key="your_genai_api_key_here"
```

Replace your_openai_api_key_here and your_genai_api_key_here with your actual API keys.
Prepare the data:

```bash
~$ mkdir -p ./data
```

Place your JSON data files in the ./data directory. Ensure that the files are named according to the dataset they represent, e.g., medqa.json, pubmedqa.json, etc.

Your directory structure should look like this:
```
mdagents/
├── data/
│   ├── medqa.json
│   ├── pubmedqa.json
│   └── ... (other dataset files)
├── main.py
├── utils.py
├── requirements.txt
└── README.md
```

<br>

## Dataset

<p align="center">
  <img width="900" src="imgs/datasets.png">
</p>

<br>

1) MedQA: [https://github.com/jind11/MedQA](https://github.com/jind11/MedQA)
2) MedMCQA: [https://github.com/medmcqa/medmcqa](https://github.com/medmcqa/medmcqa)
3) PubMedQA: [https://github.com/pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa)
4) DDXPlus: [https://github.com/mila-iqia/ddxplus](https://github.com/mila-iqia/ddxplus)
5) SymCat: [https://github.com/teliov/SymCat-to-synthea](https://github.com/teliov/SymCat-to-synthea)
6) JAMA & Medbullets: [https://github.com/xiaoman-zhang/PMC-VQA](https://github.com/xiaoman-zhang/PMC-VQA)
7) PMC-VQA: [https://github.com/xiaoman-zhang/PMC-VQA](https://github.com/xiaoman-zhang/PMC-VQA)
8) Path-VQA: [https://github.com/UCSD-AI4H/PathVQA](https://github.com/UCSD-AI4H/PathVQA)
9) MIMIC-CXR: [https://github.com/baeseongsu/mimic-cxr-vqa](https://github.com/baeseongsu/mimic-cxr-vqa)
10) MedVidQA: [https://github.com/deepaknlp/MedVidQACL](https://github.com/deepaknlp/MedVidQACL)

<br>

## Comparison to Previous Single Agent/Multi-Agent Methods

<p align="center">
  <img width="900" src="imgs/comparison.png">
</p>

<br>

## Inference

```bash
~$ python3 main.py --model {gpt-3.5, gpt-4, gpt-4v, gpt-4o, gemini-pro, gemini-pro-vision} --dataset {medqa, pubmedqa, ddxplus, jama, symcat, medbullets, jama, pmc-vqa, path-vqa, mimic-cxr, medvidqa}
```

<br>

## Main Results

<p align="center">
  <img width="800" alt="image" src="imgs/main_table.png">
  <img width="500" alt="image" src="imgs/radar.png">
</p>
<br>

## Ablation 1: Impact of Complexity Selection

<p align="center">
  <img width="800" alt="image" src="imgs/ablation1.png">
</p>
<br>

## Ablation 2: Impact of Number of Agents and Temperatures in Group Setting

<p align="center">
  <img width="800" alt="image" src="imgs/ablation2.png">
</p>
<br>

## Ablation 3: Impact of Moderator’s Review and RAG

<p align="center">
  <img width="600" alt="image" src="imgs/ablation4.png">
</p>
<br>



## Cite Us
If you find this repository useful in your research, please cite our paper:

```bibtex
@misc{kim2024mdagentsadaptivecollaborationllms,
      title={MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making}, 
      author={Yubin Kim and Chanwoo Park and Hyewon Jeong and Yik Siu Chan and Xuhai Xu and Daniel McDuff and Hyeonhoon Lee and Marzyeh Ghassemi and Cynthia Breazeal and Hae Won Park},
      year={2024},
      eprint={2404.15155},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.15155}, 
}
```

## Contact
Yubin Kim (ybkim95@mit.edu)
