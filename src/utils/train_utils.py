import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class HierarchicalLoss(nn.Module):
    """层次加权损失函数"""
    def __init__(self, parent_weight=1.0, child_weight=1.0, routing_weight=0.5):
        super(HierarchicalLoss, self).__init__()
        self.parent_weight = parent_weight
        self.child_weight = child_weight
        self.routing_weight = routing_weight
        self.parent_criterion = nn.CrossEntropyLoss()
        self.child_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略无效标签
    
    def forward(self, parent_logits, child_logits, routed_child_logits, parent_labels, child_labels):
        """
        计算层次损失
        
        Args:
            parent_logits: 父类预测logits
            child_logits: 子类预测logits
            routed_child_logits: 路由后的子类logits
            parent_labels: 父类真实标签
            child_labels: 子类真实标签
        
        Returns:
            total_loss: 总损失
            parent_loss: 父类损失
            child_loss: 子类损失
            routing_loss: 路由损失
        """
        # 确保标签维度正确
        if isinstance(parent_labels, tuple):
            parent_labels = parent_labels[0]
        if isinstance(child_labels, tuple):
            child_labels = child_labels[0]
        
        # 转换标签为一维
        if len(parent_labels.shape) > 1:
            parent_labels = parent_labels.view(-1)
        if len(child_labels.shape) > 1:
            child_labels = child_labels.view(-1)
            
        # 父类损失
        parent_loss = self.parent_criterion(parent_logits, parent_labels)
        
        # 子类损失
        child_loss = self.child_criterion(child_logits, child_labels)
        
        # 路由子类损失
        routing_loss = self.child_criterion(routed_child_logits, child_labels)
        
        # 总损失
        total_loss = (self.parent_weight * parent_loss + 
                      self.child_weight * child_loss + 
                      self.routing_weight * routing_loss)
        
        return total_loss, parent_loss, child_loss, routing_loss


def train_epoch(model, data_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    parent_losses = 0
    child_losses = 0
    routing_losses = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        # 将数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        tfidf_features = batch['tfidf_features'].to(device)
        malicious_features = batch['malicious_features'].to(device)
        parent_labels = batch['parent_label'].to(device)
        child_labels = batch['child_label'].to(device)
        
        # 前向传播
        parent_logits, child_logits, routed_child_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            tfidf_features=tfidf_features,
            malicious_features=malicious_features,
            parent_label=parent_labels
        )
        
        # 计算损失
        outputs = (parent_logits, child_logits, routed_child_logits)
        targets = (parent_labels, child_labels)
        loss, parent_loss, child_loss, routing_loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累加损失
        total_loss += loss.item()
        parent_losses += parent_loss.item()
        child_losses += child_loss.item()
        routing_losses += routing_loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': loss.item(),
            'p_loss': parent_loss.item(),
            'c_loss': child_loss.item(),
            'r_loss': routing_loss.item()
        })
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    avg_parent_loss = parent_losses / len(data_loader)
    avg_child_loss = child_losses / len(data_loader)
    avg_routing_loss = routing_losses / len(data_loader)
    
    return avg_loss, avg_parent_loss, avg_child_loss, avg_routing_loss


def evaluate(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    parent_losses = 0
    child_losses = 0
    routing_losses = 0
    
    parent_preds = []
    parent_true = []
    child_preds = []
    child_true = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            tfidf_features = batch['tfidf_features'].to(device)
            malicious_features = batch['malicious_features'].to(device)
            parent_labels = batch['parent_label'].to(device)
            child_labels = batch['child_label'].to(device)
            
            # 前向传播
            parent_logits, child_logits, routed_child_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                tfidf_features=tfidf_features,
                malicious_features=malicious_features
            )
            
            # 计算损失
            outputs = (parent_logits, child_logits, routed_child_logits)
            targets = (parent_labels, child_labels)
            loss, parent_loss, child_loss, routing_loss = criterion(outputs, targets)
            
            # 累加损失
            total_loss += loss.item()
            parent_losses += parent_loss.item()
            child_losses += child_loss.item()
            routing_losses += routing_loss.item()
            
            # 获取预测结果
            parent_pred = torch.argmax(parent_logits, dim=1).cpu().numpy()
            parent_preds.extend(parent_pred)
            parent_true.extend(parent_labels.cpu().numpy())
            
            child_pred = torch.argmax(routed_child_logits, dim=1).cpu().numpy()
            child_preds.extend(child_pred)
            child_true.extend(child_labels.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    avg_parent_loss = parent_losses / len(data_loader)
    avg_child_loss = child_losses / len(data_loader)
    avg_routing_loss = routing_losses / len(data_loader)
    
    # 计算父类评估指标
    parent_true = np.array(parent_true).flatten()
    parent_preds = np.array(parent_preds).flatten()
    parent_acc = accuracy_score(parent_true, parent_preds)
    parent_precision, parent_recall, parent_f1, _ = precision_recall_fscore_support(
        parent_true, parent_preds, average='weighted'
    )
    
    # 计算子类评估指标（忽略无效标签）
    valid_indices = np.where(np.array(child_true).flatten() != -1)[0]
    if len(valid_indices) > 0:
        child_true_valid = np.array(child_true).flatten()[valid_indices]
        child_preds_valid = np.array(child_preds).flatten()[valid_indices]
        child_acc = accuracy_score(child_true_valid, child_preds_valid)
        child_precision, child_recall, child_f1, _ = precision_recall_fscore_support(
            child_true_valid, child_preds_valid, average='weighted'
        )
    else:
        child_acc = child_precision = child_recall = child_f1 = 0.0
    
    # 返回评估结果
    eval_results = {
        'loss': avg_loss,
        'parent_loss': avg_parent_loss,
        'child_loss': avg_child_loss,
        'routing_loss': avg_routing_loss,
        'parent_acc': parent_acc,
        'parent_precision': parent_precision,
        'parent_recall': parent_recall,
        'parent_f1': parent_f1,
        'child_acc': child_acc,
        'child_precision': child_precision,
        'child_recall': child_recall,
        'child_f1': child_f1
    }
    
    return eval_results


def predict(model, data_loader, device):
    """预测"""
    model.eval()
    parent_preds = []
    child_preds = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            tfidf_features = batch['tfidf_features'].to(device)
            malicious_features = batch['malicious_features'].to(device)
            
            # 前向传播
            parent_logits, _, routed_child_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                tfidf_features=tfidf_features,
                malicious_features=malicious_features
            )
            
            # 获取预测结果
            parent_pred = torch.argmax(parent_logits, dim=1).cpu().numpy()
            parent_preds.extend(parent_pred)
            
            child_pred = torch.argmax(routed_child_logits, dim=1).cpu().numpy()
            child_preds.extend(child_pred)
    
    return np.array(parent_preds), np.array(child_preds) 