# LCGE

## 简介
这是 [LCGE](https://arxiv.org/pdf/2211.16865.pdf) 框架的 PyTorch 实现。我们提出了一个逻辑和常识引导的嵌入模型（LCGE），以共同学习涉及事件的及时性和因果关系的时态表示，以及从常识角度看事件的无时态表示。具体而言，我们设计了一个时态规则学习算法，构建了一个规则引导的谓词嵌入正则化策略，用于学习事件之间的因果关系。此外，我们可以通过辅助常识知识准确评估事件的合理性。

## 安装
创建一个包含 PyTorch 和 scikit-learn 的 conda 环境：
```bash
conda create --name lcge_env python=3.7
source activate lcge_env
conda install --file requirements.txt -c pytorch
```
然后将 lcge 包安装到此环境：
```bash
python setup.py install
```

## 数据集
通过运行以下命令处理数据集并将其添加到包数据文件夹中：
```bash
cd lcge
python process_icews.py
python process_wikidata12k.py
```
全局静态知识图可以通过 `./src_data/static_data` 文件夹中的 `staticgraph_icews.ipynb` 构建。

要生成静态规则，可以使用 `./src_data/rulelearning` 文件夹中的 AMIE+ 工具 `amie_plus.jar`。

可以通过 `./src_data/rulelearning` 文件夹中的 `temporal_rule_learning_icews14.ipynb` 和 `temporal_rule_learning_icews15.ipynb` 生成时态规则。

`triples.tsv`：生成的全局静态知识图。
`rule1_p1.json` 和 `rule1_p2.json`：长度为1的时态规则。
`rule2_p1.txt`，`rule2_p2.txt`，`rule2_p3.txt` 和 `rule2_p4.txt`：长度为2的时态规则。

## 训练和测试
为了复现 LCGE 模型在数据集上的结果，您可以运行以下命令：
**ICEWS14:**
```bash
python learner_lcge.py --dataset ICEWS14 --model LCGE --rank 2000 --emb_reg 0.005 --time_reg 0.01 --rule_reg 0.01 --max_epoch 1000 --weight_static 0.1 --learning_rate 0.1
```

**ICEWS05-15:**
```bash
python learner_lcge.py --dataset ICEWS05-15 --model LCGE --rank 2000 --emb_reg 0.0025 --time_reg 0.05 --rule_reg 1.0 --max_epoch 1000 --weight_static 0.1 --learning_rate 0.1
```

**Wikidata12k:**
```bash
python learner_cs.py --dataset wikidata12k --model LCGE --rank 2000 --emb_reg 0.2 --time_reg 0.5 --max_epoch 500 --weight_static 0.1 --learning_rate 0.1
```

