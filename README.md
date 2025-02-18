# CoLLMLight: Cooperative Large Language Model Agents for Network-Wide Traffic Signal Control

<a id="requirements"></a>
## Requirements

`python>=3.9`,`tensorflow-cpu=2.8.0`, `cityflow`, `pandas=1.5.0`, `numpy=1.26.2`, `wandb`,  `transformers=4.48.2`, `vllm`, `lmdeploy`

[`cityflow`](https://github.com/cityflow-project/CityFlow.git) needs a linux environment, and we run the code on Ubuntu.

<a id="Quick Start"></a>
## Quick Start

First, deploy a LLM server through `lmdeploy`
```shell
lmdeploy serve api_server YOUR_LLM_PATH --tp=YOUR_GPU_NUM
```
Then, run the CoLLMLight
```shell
python run_CoLLMlight.py --model_path=YOUR_LLM_PATH --dataset='newyork_28x7' --traffic_file='anon_28_7_newyork_real_double.json'
```
