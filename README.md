# Adaptive Collaboration Strategy for LLMs in Medical Decision Making (2024)

<!--<p align="center">
   <img width="1100" alt="image" src="https://github.com/mitmedialab/MDAgents/assets/45308022/d08d666a-ccae-4cd7-ab5b-f2f61e769ab1">
</p>-->

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

> [!CAUTION]
> main.py will be updated soon to the latest version. 

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

<br>


## TODO

- [ ] update the main table with gpt-4o
- [ ] update main.py with the latest version
- [ ] add baseline methods
- [ ] add eval.py
- [ ] add more benchmarks (medmcqa, mmlu, mmmu, inspect, etc)
