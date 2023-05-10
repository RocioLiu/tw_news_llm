# tw_news_llm
Parameter-efficient fine-tuning ( [LoRA](https://arxiv.org/abs/2106.09685) ) [bigscience/bloomz-1b7](https://huggingface.co/bigscience/bloomz-1b7) for the news of 27 different Taiwan public sectors.

## Quickstart
### Prerequisites
#### virtualenv option
* Create a python virtual environment `virtualenv venv`
* Source `source venv/bin/activate`

#### conda option
* Create a python virtual environment 
* `conda create --name venv python=3.9`
* Source `conda activate venv`

### Installing
* Install required python package `pip install -r requirements.txt`

### Data preprocess
* Download the json data from [https://data.gov.tw/](https://data.gov.tw/). The details are listed [here](https://docs.google.com/spreadsheets/d/17GSYb3TiYKJqqIc7k4HnpcQTSd3rpcIw4Y7H4VvhU44/edit?usp=sharing)
* run the following command from data preprocessing
```bash
python tw_news_llm/data_preprocess.py
```
* Place the generated data files in `data` folder.

### Training
* Run the following command for model training
```bash
python tw_news_llm/run_pretrain_twnewsllm.py
```
