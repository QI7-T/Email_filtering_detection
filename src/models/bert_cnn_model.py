import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from data.malicious_feature_extractor import MaliciousFeatureNetwork

class AttentionLayer(nn.Module):
    """注意力层，用于CNN特征加权"""
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        context = torch.sum(attention_weights * x, dim=1)  # [batch_size, hidden_dim]
        return context, attention_weights


class MultiScaleCNN(nn.Module):
    """多尺度CNN特征提取器"""
    def __init__(self, input_dim, hidden_dim, kernel_sizes=[3, 5, 7], dropout=0.2):
        super(MultiScaleCNN, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [batch_size, 1, seq_len]
        
        # 多尺度卷积
        conv_outputs = []
        for conv in self.convs:
            # 卷积
            conv_out = conv(x)  # [batch_size, hidden_dim, seq_len]
            # 最大池化
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # [batch_size, hidden_dim]
            conv_outputs.append(pooled)
        
        # 拼接多尺度特征
        multi_scale_features = torch.cat(conv_outputs, dim=1)  # [batch_size, hidden_dim * len(kernel_sizes)]
        
        # Dropout
        multi_scale_features = self.dropout(multi_scale_features)
        
        return multi_scale_features


class ParentClassifier(nn.Module):
    """父类分类器"""
    def __init__(self, input_dim, num_classes=2):
        super(ParentClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)


class ChildClassifier(nn.Module):
    """子类分类器"""
    def __init__(self, input_dim, num_classes=3):
        super(ChildClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),  # 子类分类器使用更高的dropout
            nn.Linear(input_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)


class BertCNNHierarchicalModel(nn.Module):
    """BERT-CNN双层级联模型"""
    def __init__(self, bert_model_path, tfidf_dim=20000, bert_hidden_dim=768, 
                 cnn_hidden_dim=256, num_parent_classes=2, num_child_classes=3,
                 freeze_bert=True, malicious_feature_dim=14, malicious_hidden_dim=32):
        super(BertCNNHierarchicalModel, self).__init__()
        
        # BERT模型
        self.bert = BertModel.from_pretrained(bert_model_path)
        
        # 冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 保存TF-IDF维度，用于后续检查
        self.tfidf_dim = tfidf_dim
        
        # CNN分支
        self.cnn = MultiScaleCNN(
            input_dim=tfidf_dim,
            hidden_dim=cnn_hidden_dim,
            kernel_sizes=[3, 5, 7],
            dropout=0.2
        )
        
        # 恶意邮件特征网络
        self.malicious_feature_network = MaliciousFeatureNetwork(
            input_dim=malicious_feature_dim,
            hidden_dim=malicious_hidden_dim
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(bert_hidden_dim + cnn_hidden_dim * 3 + malicious_hidden_dim, bert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 父类分类器
        self.parent_classifier = ParentClassifier(
            input_dim=bert_hidden_dim,
            num_classes=num_parent_classes
        )
        
        # 子类分类器
        self.child_classifier = ChildClassifier(
            input_dim=bert_hidden_dim,
            num_classes=num_child_classes
        )
        
        # 路由门控机制 - 只有垃圾邮件才有子类
        self.routing_gates = nn.ModuleDict({
            'spam': nn.Linear(bert_hidden_dim, 3)  # 垃圾邮件子类（恶意、广告、欺诈）
        })
        
    def unfreeze_bert_layers(self, num_layers=3):
        """解冻BERT顶层的指定层数"""
        # 首先确保所有参数都被冻结
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # 解冻顶层encoder
        for i in range(12 - num_layers, 12):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
                
        # 解冻pooler
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, token_type_ids, tfidf_features, malicious_features, parent_label=None):
        """
        前向传播
        
        Args:
            input_ids: BERT输入ID
            attention_mask: BERT注意力掩码
            token_type_ids: BERT token类型ID
            tfidf_features: TF-IDF特征
            malicious_features: 恶意邮件特征
            parent_label: 父类标签（用于训练时的路由）
            
        Returns:
            parent_logits: 父类预测logits
            child_logits: 子类预测logits
            routed_child_logits: 基于父类路由后的子类logits
        """
        # BERT特征提取
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_features = bert_outputs.pooler_output  # [batch_size, bert_hidden_dim]
        
        # 检查并处理TF-IDF特征维度
        batch_size, actual_tfidf_dim = tfidf_features.shape
        if actual_tfidf_dim != self.tfidf_dim:
            # 如果维度不匹配，使用零填充或截断
            if actual_tfidf_dim < self.tfidf_dim:
                # 零填充
                padding = torch.zeros(batch_size, self.tfidf_dim - actual_tfidf_dim, device=tfidf_features.device)
                tfidf_features = torch.cat([tfidf_features, padding], dim=1)
            else:
                # 截断
                tfidf_features = tfidf_features[:, :self.tfidf_dim]
        
        # CNN特征提取
        cnn_features = self.cnn(tfidf_features.unsqueeze(1))  # [batch_size, cnn_hidden_dim*3]
        
        # 恶意邮件特征提取
        malicious_net_features = self.malicious_feature_network(malicious_features)  # [batch_size, malicious_hidden_dim]
        
        # 特征融合
        combined_features = torch.cat([bert_features, cnn_features, malicious_net_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # 父类预测
        parent_logits = self.parent_classifier(fused_features)
        
        # 子类预测
        child_logits = self.child_classifier(fused_features)
        
        # 初始化路由后的子类logits
        routed_child_logits = torch.zeros_like(child_logits)
        
        # 基于父类标签的路由
        if parent_label is not None:
            # 训练时使用真实的父类标签进行路由
            for i, p_label in enumerate(parent_label.squeeze()):
                # 只有垃圾邮件(索引1)才有子类
                if p_label == 1:  # spam
                    gate_logits = self.routing_gates['spam'](fused_features[i:i+1])
                    routed_child_logits[i] = gate_logits
        else:
            # 推理时使用预测的父类标签进行路由
            _, parent_preds = torch.max(parent_logits, dim=1)
            for i, p_label in enumerate(parent_preds):
                # 只有垃圾邮件(索引1)才有子类
                if p_label == 1:  # spam
                    gate_logits = self.routing_gates['spam'](fused_features[i:i+1])
                    routed_child_logits[i] = gate_logits
        
        return parent_logits, child_logits, routed_child_logits 