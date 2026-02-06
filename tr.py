import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold
from collections import Counter
import csv
import json
import itertools

# 导入自定义模块
from net.st_gcn import Model

# 数据集类
class Feeder(Dataset):
    def __init__(self, data_path, label_path, with_filename=False):
        self.data = np.load(data_path, allow_pickle=True)
        with open(label_path, 'rb') as f:
            data = pickle.load(f)
            self.filenames = data[0]  # 文件名列表
            self.labels = data[1]  # 标签列表
        self.with_filename = with_filename

    def __getitem__(self, index):
        if self.with_filename:
            filename = self.filenames[index]
            return torch.from_numpy(self.data[index]).float(), torch.tensor(self.labels[index], dtype=torch.long), filename
        return torch.from_numpy(self.data[index]).float(), torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)

# 初始化模型权重
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# 动态计算类别权重
def compute_class_weights(labels):
    class_count = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_count)
    class_weights = {cls: total_samples / (num_classes * count) for cls, count in class_count.items()}
    return class_weights

# 动态计算alpha参数
def compute_alpha(labels):
    class_count = Counter(labels)
    alpha = class_count[1] / class_count[0]
    alpha = 0.25
    return alpha

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DynamicMarginTripletCenterLoss(nn.Module):
    def __init__(self, initial_margin=1.0, margin_decay=0.99):
        super(DynamicMarginTripletCenterLoss, self).__init__()
        self.initial_margin = initial_margin
        self.margin_decay = margin_decay
        self.current_margin = initial_margin

    def forward(self, anchor, positive, negative, epoch):
        # 动态调整margin
        self.current_margin = self.initial_margin * (self.margin_decay ** epoch)

        pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=1)
        neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=1)
        loss = F.relu(pos_dist - neg_dist + self.current_margin)
        return loss.mean()

def hard_triplet_mining(anchor, positive, negative):
    # 计算余弦距离
    pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=-1)

    # 选择困难三元组
    hard_triplets = pos_dist > neg_dist

    # 通过广播和重塑来匹配输入张量的形状
    hard_triplets = hard_triplets.unsqueeze(-1).expand_as(anchor)

    return anchor[hard_triplets].view(-1, anchor.size(-1)), \
           positive[hard_triplets].view(-1, positive.size(-1)), \
           negative[hard_triplets].view(-1, negative.size(-1))

# 训练参数
NUM_EPOCH = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001  # 调整初始学习率
WEIGHT_DECAY = 1e-4
PATIENCE = 4  # 提前停止的耐心值

# 数据集加载
data_path = r'D:\infant program\16\train_data.npy'
label_path = r'D:\infant program\16\rain_label.pkl'
dataset = Feeder(data_path=data_path, label_path=label_path, with_filename=True)

print("Dataset loaded successfully.")

# K折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 记录性能指标
all_train_accuracies = []
all_test_accuracies = []
all_precisions = []
all_recalls = []
all_f1_scores = []
all_confusion_matrices = []
all_roc_curves = []
all_auc_scores = []
all_sensitivities = []
all_specificities = []

# 设置结果保存路径
results_folder = r'D:\infant program\st-gcn-master\result\16-15'
os.makedirs(results_folder, exist_ok=True)

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f'Starting Fold {fold + 1}')

    # 创建每个fold的结果文件夹
    fold_folder = os.path.join(results_folder, f'fold_{fold+1}')
    os.makedirs(fold_folder, exist_ok=True)

    # 数据加载器
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train set size: {len(train_idx)}, Test set size: {len(test_idx)}")

    # 动态计算当前折的类别权重和alpha值
    train_labels = [dataset[i][1].item() for i in train_idx]
    class_weights_dict = compute_class_weights(train_labels)
    class_weights = torch.FloatTensor([class_weights_dict[i] for i in range(len(class_weights_dict))]).to(device)

    alpha = compute_alpha(train_labels)

    print(f"Class weights: {class_weights}")
    print(f"Alpha value: {alpha}")

    # 模型和优化器

    model = Model(
        in_channels=6,  # 假设输入通道数为6
        num_class=2,
        dropout=0.6,
        edge_importance_weighting=True,
        graph_args={'layout': 'mmpose', 'strategy': 'spatial'}
    ).to(device)

    model.apply(initialize_weights)
    print('debug', model.graph_unet.proj.weight.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    print("Model initialized and moved to device.")

    best_train_acc = 0.0
    best_test_acc = 0.0
    best_model_state = None
    best_train_epoch = 0
    best_test_epoch = 0
    best_metrics = {'train': {}, 'test': {}}
    best_outputs = []

    focal_loss = FocalLoss(weight=class_weights).to(device)
    triplet_center_loss = DynamicMarginTripletCenterLoss(initial_margin=1.0, margin_decay=0.9).to(device)

    # 创建CSV文件来记录每轮的结果
    csv_file = os.path.join(fold_folder, 'training_results.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Precision', 'F1 Score',
                         'Sensitivity', 'Specificity', 'AUROC'])

    train_losses = []
    test_losses = []

    early_stopping_counter = 0
    best_test_loss = float('inf')

    for epoch in range(NUM_EPOCH):
        model.train()
        total_loss = 0
        correct_train = 0
        y_true = []
        y_pred = []

        for data in train_loader:
            inputs, labels, filenames = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            # 假设我们有 anchor, positive, negative 样本
            anchor, positive, negative = inputs[:, 0, :, :], inputs[:, 1, :, :], inputs[:, 2, :, :]
            anchor, positive, negative = hard_triplet_mining(anchor, positive, negative)

            focal_loss_value = focal_loss(output, labels)
            triplet_center_loss_value = triplet_center_loss(anchor, positive, negative, epoch)

            # 总损失
            loss = focal_loss_value + triplet_center_loss_value
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)  # 梯度裁剪
            optimizer.step()
            total_loss += loss.item()

            pred = torch.argmax(output, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(pred.tolist())
            correct_train += (pred == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_train / len(train_loader.dataset)
        train_losses.append(train_loss)

        if train_accuracy > best_train_acc:
            best_train_acc = train_accuracy
            best_train_epoch = epoch + 1
            best_metrics['train'] = {
                'loss': train_loss,
                'accuracy': train_accuracy,
                'precision': precision_score(y_true, y_pred,  zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0)
            }

        # 测试循环
        model.eval()
        correct_test = 0
        y_true_test = []
        y_pred_test = []
        y_scores = []
        total_test_loss = 0
        current_outputs = []

        with torch.no_grad():
            for data in test_loader:
                inputs, labels, filenames = data
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)

                loss = focal_loss(output, labels)
                total_test_loss += loss.item()

                pred = torch.argmax(output, dim=1)
                correct_test += (pred == labels).sum().item()

                y_true_test.extend(labels.tolist())
                y_pred_test.extend(pred.tolist())
                y_scores.extend(output[:, 1].cpu().numpy())  # 假设正类的概率在第二列

                # 保存当前输出和对应的文件名
                current_outputs.extend(zip(filenames, output.cpu().numpy()))

        test_loss = total_test_loss / len(test_loader)
        test_accuracy = correct_test / len(test_loader.dataset)
        test_losses.append(test_loss)
        precision_test = precision_score(y_true_test, y_pred_test, zero_division=0)
        recall_test = recall_score(y_true_test, y_pred_test, zero_division=0)
        f1_test = f1_score(y_true_test, y_pred_test, zero_division=0)

        # 计算灵敏度和特异性
        conf_matrix = confusion_matrix(y_true_test, y_pred_test)
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fpr, tpr, _ = roc_curve(y_true_test, y_scores)
        roc_auc = auc(fpr, tpr)

        print(
            f'Epoch {epoch + 1}/{NUM_EPOCH}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Precision: {precision_test:.4f}, F1: {f1_test:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, AUROC: {roc_auc:.4f}')

        # 将结果写入CSV文件
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [epoch + 1, train_loss, train_accuracy, test_loss, test_accuracy, precision_test, recall_test, f1_test,
                 sensitivity, specificity, roc_auc])

        # 如果测试准确率提高，则保存模型和分类错误的文件名
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_test_epoch = epoch + 1
            best_model_state = model.state_dict()
            best_metrics['test'] = {
                'loss': test_loss,
                'accuracy': test_accuracy,
                'precision': precision_test,
                'f1': f1_test,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'auroc': roc_auc
            }
            best_outputs = current_outputs

        # 提前停止逻辑
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # 调整学习率
        scheduler.step(test_loss)

    print(f'\nFold {fold + 1} Complete')
    print(f'Best Train Accuracy: {best_train_acc:.4f} at epoch {best_train_epoch}')
    print(f'Best Test Accuracy: {best_test_acc:.4f} at epoch {best_test_epoch}')
    print('Best Train Metrics:', best_metrics['train'])
    print('Best Test Metrics:', best_metrics['test'])
    print('=' * 50)

    # 保存最佳模型
    torch.save(best_model_state, os.path.join(fold_folder, f'best_model_fold_{fold + 1}.pth'))

    # 保存最佳输出和文件名
    with open(os.path.join(fold_folder, f'best_outputs_fold_{fold + 1}.txt'), 'w') as f:
        for filename, output in best_outputs:
            f.write(f"{filename}: {output}\n")

    # 记录实际训练的 epoch 数量
    actual_epochs = len(train_losses)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, actual_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves - Fold {fold + 1}')
    plt.legend()
    plt.savefig(os.path.join(fold_folder, f'loss_curves_fold_{fold + 1}.png'))
    plt.close()

    # 记录这个fold的最佳性能
    all_train_accuracies.append(best_metrics['train']['accuracy'])
    all_test_accuracies.append(best_metrics['test']['accuracy'])
    all_precisions.append(best_metrics['test']['precision'])
    all_f1_scores.append(best_metrics['test']['f1'])
    all_sensitivities.append(best_metrics['test']['sensitivity'])
    all_specificities.append(best_metrics['test']['specificity'])
    all_auc_scores.append(best_metrics['test']['auroc'])

    # 记录混淆矩阵和ROC曲线数据
    all_confusion_matrices.append(conf_matrix)
    all_roc_curves.append((fpr, tpr, roc_auc))

# 计算平均性能
avg_train_accuracy = np.mean(all_train_accuracies)
avg_test_accuracy = np.mean(all_test_accuracies)
avg_precision = np.mean(all_precisions)
avg_recall = np.mean(all_recalls)
avg_f1_score = np.mean(all_f1_scores)
avg_sensitivity = np.mean(all_sensitivities)
avg_specificity = np.mean(all_specificities)
avg_auroc = np.mean(all_auc_scores)

print("\nOverall Results:")
print(f"Average Train Accuracy: {avg_train_accuracy:.4f}")
print(f"Average Test Accuracy: {avg_test_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average F1 Score: {avg_f1_score:.4f}")
print(f"Average Sensitivity: {avg_sensitivity:.4f}")
print(f"Average Specificity: {avg_specificity:.4f}")
print(f"Average AUROC: {avg_auroc:.4f}")

# 保存总体结果
overall_results = {
    'avg_train_accuracy': avg_train_accuracy,
    'avg_test_accuracy': avg_test_accuracy,
    'avg_precision': avg_precision,
    'avg_recall': avg_recall,
    'avg_f1_score': avg_f1_score,
    'avg_sensitivity': avg_sensitivity,
    'avg_specificity': avg_specificity,
    'avg_auroc': avg_auroc
}



with open(os.path.join(results_folder, 'overall_results.json'), 'w') as f:
    json.dump(overall_results, f, indent=4)

# 计算平均混淆矩阵
avg_conf_matrix = np.mean(all_confusion_matrices, axis=0)

plt.figure(figsize=(8, 6))
plt.imshow(avg_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Average Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Class 0', 'Class 1'], rotation=45)
plt.yticks(tick_marks, ['Class 0', 'Class 1'])

thresh = avg_conf_matrix.max() / 2.
for i, j in itertools.product(range(avg_conf_matrix.shape[0]), range(avg_conf_matrix.shape[1])):
    plt.text(j, i, format(avg_conf_matrix[i, j], '.2f'),
             horizontalalignment="center",
             color="white" if avg_conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(results_folder, 'average_confusion_matrix.png'))
plt.close()

# 插值并计算平均ROC曲线
mean_fpr = np.linspace(0, 1, 100)
tprs = []

for fpr, tpr, _ in all_roc_curves:
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.figure(figsize=(8, 6))
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUROC = {mean_auc:.2f})', lw=2, alpha=.8)

for fpr, tpr, roc_auc in all_roc_curves:
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold (AUROC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(results_folder, 'average_roc_curve.png'))
plt.close()

print("All results have been saved successfully.")
