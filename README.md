# Zero-shot-learning-for-text-to-SQL
## MC—SQL
### Prepare Data
Download pre-trained BERT model [bert-base-uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) and [bert-base-chinese](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz), unzip and put them under MC-SQL/data/.
### Running Code
1. Preprocess data for perform Coarse-grained Filtering  
`cd ./src/preprocess`  
`python enhance_header_wikisql.py`  
`cd ..`
2. Training
Then, execute the following command for training MC-SQL.  
`sh train_wiksql.sh`


## T5
### Prepare Data
Download [pre-trained model](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) unzip and put them under t5-demo/.
### Running Code
1. Training  
`cd /t5-demo`  
`python main.py`  

2. Demo test
Then, execute the following command for demo t5.  
`python demo.py`
