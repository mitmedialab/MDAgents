# Adaptive Collaboration Strategy for LLMs in Medical Decision Making (2024)

Foundation models have become invaluable in advancing the medical field. Despite their promise, the strategic deployment of LLMs for effective utility in complex medical tasks remains an open question. Our novel framework, Medical Decision-making Agents (**MDAgents**) aims to address this gap by automatically assigning the effective collaboration structure for LLMs. Assigned solo or group collaboration structure is tailored to the complexity of the medical task at hand, emulating real-world medical decision making processes. We evaluate our framework and baseline methods with state-of-the-art LLMs across a suite of challenging medical benchmarks: MedQA, MedMCQA, PubMedQA, DDXPlus, PMC-VQA, Path-VQA, and MedVidQA, achieving the best performance in **5 out of 7** benchmarks that require an understanding of multi-modal medical reasoning. Ablation studies reveal that MDAgents excels in adapting the number of collaborating agents to optimize efficiency and accuracy, showcasing its robustness in diverse scenarios. We also explore the dynamics of group consensus, offering insights into how collaborative agents could behave in complex clinical team dynamics.

![new_framework](https://github.com/mitmedialab/MDAgents/assets/45308022/46f9b29c-904b-4875-925b-f6edd66de4ee)

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

1) MedQA: [https://github.com/jind11/MedQA?tab=readme-ov-file](https://github.com/jind11/MedQA?tab=readme-ov-file)
2) MedMCQA: [https://github.com/medmcqa/medmcqa](https://github.com/medmcqa/medmcqa)
3) PubMedQA: [https://github.com/pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa)
4) DDXPlus: [https://github.com/mila-iqia/ddxplus](https://github.com/mila-iqia/ddxplus)
5) PMC-VQA: [https://github.com/xiaoman-zhang/PMC-VQA](https://github.com/xiaoman-zhang/PMC-VQA)
6) Path-VQA: [https://github.com/UCSD-AI4H/PathVQA](https://github.com/UCSD-AI4H/PathVQA)
7) MedVidQA: [https://github.com/deepaknlp/MedVidQACL](https://github.com/deepaknlp/MedVidQACL)

<br>

## Inference

```bash
~$ python3 main.py --model {gpt-3.5, gpt-4, gpt-4v, gemini-pro, gemini-pro-vision} --dataset {medqa, medmcqa, pubmedqa, ddxplus, pmc-vqa, path-vqa, medvidqa}
```

<br>

If our work is helpful to you, please kindly cite our paper as:

```
@article{kim2024adaptive,
  title={Adaptive Collaboration Strategy for LLMs in Medical Decision Making},
  author={Kim, Yubin and Park, Chanwoo and Jeong, Hyewon and Chan, Yik Siu and Xu, Xuhai and McDuff, Daniel and Breazeal, Cynthia and Park, Hae Won},
  journal={arXiv preprint arXiv:2404.15155},
  year={2024}
}
```

## TODO

- [ ] add baseline models
- [ ] add eval.py
- [ ] add more benchmarks
