# BERT-CNN层次细粒度邮件分类系统

本项目实现了一个基于BERT-CNN双层级联的邮件层次细粒度分类过滤检测系统，能够将邮件分为多个层次的类别，有效识别各类垃圾邮件。

## 项目结构

```
.
├── bert-base-chinese/      # BERT预训练模型
├── trec06c/               # 邮件数据集
├── src/                   # 源代码
│   ├── data/              # 数据处理模块
│   │   ├── data_processor.py     # 数据预处理
│   │   ├── data_augmentation.py  # 数据增强
│   │   └── malicious_feature_extractor.py  # 恶意特征提取
│   ├── models/            # 模型定义
│   │   └── bert_cnn_model.py     # BERT-CNN模型
│   ├── utils/             # 工具函数
│   │   ├── train_utils.py        # 训练工具
│   │   ├── visualization.py      # 可视化工具
│   │   └── word_cloud.py         # 词云生成
│   ├── train.py           # 训练脚本
│   └── predict.py         # 预测脚本
├── simhei.ttf             # 中文字体文件
├── requirements.txt       # 项目依赖
└── README.md              # 项目说明
```

## 模型架构

该模型采用BERT-CNN双层级联架构，实现邮件的层次细粒度分类：

1. **父级分类（粗粒度）**：
   - 将邮件分为正常邮件、垃圾邮件两个父类
   - 结合BERT的语义理解能力和CNN的局部特征提取能力

2. **子级分类（细粒度）**：
  将垃圾邮件细分为恶意邮件、广告邮件、欺诈邮件，其中各种邮件所含内容如下：
   - 恶意邮件：网络钓鱼邮件、可疑网站邮件
   - 广告邮件：健康医疗邮件、商业金融邮件、学术会议邮件
   - 欺诈邮件：博彩邮件、色情邮件、虚假违法服务邮件

3. **关键技术点**：
   - BERT分支：使用预训练的中文BERT模型捕获全文语义
   - CNN分支：使用多尺度卷积核（3/5/7-gram）并行提取复杂局部模式
   - 注意力机制：在CNN分支添加注意力模块，增强关键词特征
   - 动态路由：根据父类预测结果，激活对应的子分类器网络
   - 层次加权损失：综合父类分类损失、子类分类损失和路由损失

4. **特征融合**：
   - BERT特征：捕获语义级别的理解
   - TF-IDF + CNN特征：捕获关键词和局部模式
   - 恶意邮件特征：专门针对恶意邮件的特征提取

## 环境要求

- Python 3.7+
- PyTorch 1.8.0+
- Transformers 4.5.0+
- scikit-learn 0.24.0+
- pandas 1.2.0+
- numpy 1.19.0+
- matplotlib 3.3.0+
- tqdm 4.50.0+
- jieba 0.42.0+
- tensorboard 2.4.0+
- wordcloud 1.8.0+

可以通过以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集

使用trec06c邮件数据集，该数据集包含多种类型的邮件，每个邮件包括邮件头和邮件正文内容。本模型只使用邮件正文内容进行分类。

数据集结构：
- `trec06c/ham/`: 正常邮件
- `trec06c/Malicious emails/`: 恶意邮件
  - `Phishing/`: 网络钓鱼邮件
  - `Service advocacy/`: 可疑网站邮件
- `trec06c/Advertising emails/`: 广告邮件
  - `Healthcare/`: 健康医疗邮件
  - `Business Finance/`: 商业金融邮件
  - `Academic education/`: 学术会议邮件
- `trec06c/Scam emails/`: 欺诈邮件
  - `Gambling merchandising/`: 博彩邮件
  - `Erotic seduction/`: 色情邮件
  - `illegal services/`: 虚假违法服务邮件

## 使用方法

### 训练模型

```bash
python src/train.py --use_augmentation --malicious_weight 1.5 --data_dir trec06c 
--bert_model_path bert-base-chinese --output_dir outputs --batch_size 16 --epochs 10 
--freeze_bert --unfreeze_epoch 3 --unfreeze_layers 3 --use_gpu
```

主要参数说明：
- `--use_augmentation`: 启用数据增强，特别是对恶意邮件进行过采样
- `--malicious_weight 1.5`: 设置恶意邮件子类的损失权重为1.5倍
- `--data_dir`: 数据集目录
- `--bert_model_path`: BERT预训练模型路径
- `--output_dir`: 输出目录
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数
- `--freeze_bert`: 是否冻结BERT（分阶段训练）
- `--unfreeze_epoch`: 解冻BERT的轮次
- `--unfreeze_layers`: 解冻BERT的层数
- `--use_gpu`: 是否使用GPU
- `--max_length`: BERT输入最大长度（默认256）
- `--tfidf_max_features`: TF-IDF最大特征数量（默认20000）
- `--parent_weight`: 父类损失权重（默认1.0）
- `--child_weight`: 子类损失权重（默认1.0）
- `--routing_weight`: 路由损失权重（默认0.5）

### 预测

```bash
python src/predict.py --model_path outputs/best_model_package.pkl --input trec06c --output 
predictions.csv --batch_size 16 --use_gpu
```

主要参数说明：
- `--model_path`: 模型包路径(.pkl文件)
- `--input`: 输入文件或目录路径
- `--output`: 输出文件路径（默认为predictions.csv）
- `--batch_size`: 批次大小（默认为16）
- `--use_gpu`: 是否使用GPU

预测完成后，系统会生成一个CSV文件，包含以下信息：
- `filename`: 邮件文件名
- `parent_class`: 父类预测结果（正常邮件、恶意邮件、广告邮件、欺诈邮件）
- `child_class`: 子类预测结果（仅垃圾邮件有子类）
- `parent_pred`: 父类预测的数值编码
- `child_pred`: 子类预测的数值编码
- `content`: 邮件内容摘要（前200字符）

同时，系统会在控制台输出分类统计信息，包括：
1. 各父类邮件数量及占比
2. 垃圾邮件子类数量及占比

## 结果可视化

训练过程中会生成以下可视化结果：

1. 训练和验证损失曲线
2. 训练和验证准确率曲线
3. 父类和子类混淆矩阵
4. BERT注意力权重可视化
5. 各类邮件关键词词云
6. 特征重要性分析

结果保存在输出目录中。

## 模型性能

模型在测试集上的性能指标：

- 父类分类准确率：98%
- 父类分类F1分数：98%
- 子类分类准确率：85%
- 子类分类F1分数：85%

## 注意事项

1. 首次运行时，会自动下载BERT预训练模型（如果本地没有）
2. 推荐使用GPU进行训练，以加快训练速度
3. 模型训练时使用了分阶段训练策略，先冻结BERT，几个epoch后再解冻顶层，这有助于防止过拟合
4. 数据增强对于提高模型对少数类别的识别能力非常重要，特别是对恶意邮件类别