# ESM+: A New Evaluation Metric for Text-To-SQL

ESM+ is a new metric for the Text-to-SQL task. ESM+ calculates semantic accuracy with a lower rate of false positives than Execution accuracy and a lower rate of false negatives than Exact Set Matching. It is released along with our baselines, as well as several other state of the art model outputs. This repo contains all the code necessary for evaluation.

### Evaluation
`ESMp.py` and `esmp_process_sql.py` are written in Python 3.10, and are modeled after the [test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval).
Just like in the original evaluation scripts, to run this evaluation you need gold and predicted txt files. Examples of these are linked in [spider_dev](spider_dev/), [spider_test](spider_test/), and [cosql_dev](cosql_dev/). In each of these folders,
- `gold.txt`: gold file where each line is `gold SQL \t db_id`
- `GPT4Turbo.txt`: GPT4Turbo baseline predictions
- `Claude.txt`: Claude3Opus baseline predictions
- `C3.txt`: [C3 model](https://github.com/bigbigwatermalon/C3SQL) predictions
- `DAIL.txt`: [DAIL model](https://github.com/BeachWang/DAIL-SQL) predictions
- `DIN.txt`: [DIN model](https://github.com/MohammadrezaPourreza/Few-shot-NL2SQL-with-prompting) predictions
- `RASAT+PICARD.txt`: [RASAT+PICARD](https://github.com/LUMIA-Group/rasat) predictions
- `RESDSQL.txt`: [RESDSQL](https://github.com/RUCKBReasoning/RESDSQL) predictions
- `Graphix.txt`: [Graphix](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/graphix) predictions
- `STAR.txt`: [STAR](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/star) predictions

For the dev sets, predictions are taken directly from the corresponding githubs, with the exception of RASAT+PICARD, which was reproduced. For [spider_test](spider_test/), the predictions were reproduced using the same process as the original, but could have different results.

### Install & Run

First, download the database folders for [spider](https://drive.usercontent.google.com/download?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download&authuser=0) (dev and test) and [cosql](https://drive.usercontent.google.com/download?id=1Y3ydpFiQQ3FC0bzdfy3groV95O_f1nXF&export=download&authuser=0) (only dev).
Save the database folders into spider_dev, spider_test, and cosql_dev, respectively.

Then, create a conda environment:

```conda create -n "ESMp" python=3.10.0```
```conda activate ESMp```

Install packages:

```pip install -r requirements.txt```

To run our script, use the following command:

```python3 ESMp.py --gold path/to/gold.txt --pred path/to/pred.txt --db path/to/database/ --table path/to/tables.json```

##### Optional flags:

```--gold```: gold txt file.

```--pred```: predictions txt file.

```--db```: directory of databases.

```--table```: tables json file.

```--etype```: same as previous. Note that exe has been updated according to the paper. Default is match (ESM+).

```--plug_value```: same as previous. Note that this metric is designed for models that do predict values.

```--progress_bar_for_each_datapoint```: same as previous

```--disable_value```: add if you want to disable value checks, strongly discouraged.

```--disable_distinct```: add if you want to disable distinct checks, strongly discouraged.

```--disable_rules```: Takes a list of comma separated rules, none, or all. Rule numbers correspond to those in Table 1 of our paper. Default is none.

```--verbose```: add if you want information like which rules are being applied on each comparison.

Default configuration is to run ESM+ on spider's test set, with our baseline GPT4Turbo predictions.

### Baseline
We introduced two new baselines. These are stored in the baselines folder.

To begin, save the json files of spider and cosql you'd like to run the baselines on. They are found in the [spider](https://drive.usercontent.google.com/download?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download&authuser=0) (dev and test) and [cosql](https://drive.usercontent.google.com/download?id=1Y3ydpFiQQ3FC0bzdfy3groV95O_f1nXF&export=download&authuser=0) (only dev) datasets.

To run, first put your LLM keys in llm.py.

Then install requirements:

```pip install -r requirements.txt```

Then, baselines can be run using:

```python3 spider.py```

```python3 cosql.py```
