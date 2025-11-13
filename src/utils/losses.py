"""损失函数定义 - 专注训练优化"""
import torch
import torch.nn.functional as F


def dice_loss(y_pred, y_true, smooth=1.):
    """经典Dice损失 - 用于分割任务

    由来: 源于医学影像分割的相似性度量，直接优化分割区域重叠度
    目的: 处理类别不平衡，关注前景区域的重叠
    适用场景: 二值分割、多类别分割、医学影像
    医学影像搭配: 适合肿瘤、器官等块状区域分割

    参数:
        y_pred: 模型输出 [B, C, H, W]，应为sigmoid后的概率
        y_true: 真实标签 [B, C, H, W]，应为0-1二值
        smooth: 平滑项避免除零
    返回:
        dice_loss: 标量损失值
    """
    # 确保输入格式正确
    y_pred = y_pred.contiguous().view(y_pred.shape[0], -1)
    y_true = y_true.contiguous().view(y_true.shape[0], -1)

    intersection = (y_pred * y_true).sum(dim=1)
    union = y_pred.sum(dim=1) + y_true.sum(dim=1)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def bce_dice_loss(y_pred, y_true, dice_weight=0.5):
    """BCE + Dice组合损失 - 平衡边界和区域优化

    由来: 结合像素级分类和区域重叠优化
    目的: BCE提供精确边界，Dice处理类别不平衡
    适用场景: 需要精确边界的医学影像分割
    医学影像搭配: 器官边界、肿瘤轮廓等需要精确边缘的任务
    """
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
    dice = dice_loss(torch.sigmoid(y_pred), y_true)
    return bce + dice_weight * dice


def _soft_skeletonize(x, k=5):
    """Algorithm 1: Soft-skeletonization from the paper 可微的中心线近似提取
        实现基于迭代腐蚀的思想，使用卷积近似形态学操作（考虑使用多个尺度的腐蚀操作来近似骨架化？）
    Args:
        x: Input tensor [B, C, H, W] (probability map)
        k: Number of iterations, should be >= maximum observed radius
    Returns:
        S: Soft skeleton [B, C, H, W]
    """

    skel = torch.zeros_like(x)
    x_eroded = x.clone()

    for i in range(k):
        # I = minpool(I)
        x_eroded = -F.max_pool2d(-x_eroded, kernel_size=3, stride=1, padding=1)

        # I₀ = maxpool(minpool(I))
        x_eroded_dilated = F.max_pool2d(x_eroded, kernel_size=3, stride=1, padding=1)

        # S = S + (1 - S) * ReLU(I - I₀)
        delta_s = F.relu(x_eroded - x_eroded_dilated)
        skel = skel + (1 - skel) * delta_s

    return skel

def _topology_scores(center, surface, smooth=1e-4):
    """计算中心骨架线的包含率
    语义为软中心线精度时 - 预测中心线的包含率
    语义为软中心线敏感度时 - 真实中心线的包含率"""

    # 获取batch大小
    B = center.size(0)
    # 展平空间维度
    center_flat = center.view(B, -1)   # [B, H*W]
    surface_flat = surface.view(B, -1) # [B, H*W]

    intersection = (center_flat * surface_flat).sum(dim=1)  # [B]
    center_sum = center_flat.sum(dim=1)    # [B]
    topology_scores = (intersection + smooth) / (center_sum + smooth)  # [B]

    return topology_scores

def cl_dice_loss(pred, target, k=5):
    """经典clDice损失 - 血管分割专用

    由来: Shit et al. 2021 (https://arxiv.org/abs/2003.07311)
    目的: 专注血管拓扑结构保持，优化中心线匹配
    适用场景: 血管、神经等管状结构分割
    医学影像搭配: 视网膜血管、冠状动脉、脑血管分割

    参数:
        pred: 模型原始输出 [B, 1, H, W] (logits)
        target: 真实标签 [B, 1, H, W] (0-1二值)
        smooth: 平滑项
    返回:
        cl_dice_loss: 1 - clDice_score
    """
    smooth = 1e-4
    # 使用sigmoid获得概率图
    pred_prob = torch.sigmoid(pred)

    # 提取中心线 - 保持可微性
    pred_skeleton = _soft_skeletonize(pred_prob, k)
    target_skeleton = _soft_skeletonize(target, k)

    # 计算clDice的两个分量
    # Calculate Tprec and Tsens
    # Tprec = |S_P ⊙ V_L| / |S_P|
    tprec = _topology_scores(pred_skeleton, target)
    # Tsens = |S_L ⊙ V_P| / |S_L|
    tsens = _topology_scores(target_skeleton, pred_prob)

    # 求出cl_dice分数
    # clDice = 2 × (Tprec × Tsens) / (Tprec + Tsens)
    cl_dice = 2 * (tprec * tsens) / (tprec + tsens + smooth)
    loss = 1 - cl_dice.mean()

    return loss


def soft_cl_dice_combined_loss(pred, target, alpha=0.5, k=5):
    """Equation 3: Combined loss from the paper"""
    soft_dice_loss = dice_loss(torch.sigmoid(pred), target)
    soft_cldice_loss = cl_dice_loss(pred, target, k=k)

    loss = (1 - alpha) * soft_dice_loss + alpha * soft_cldice_loss
    return loss


def vascular_focal_loss(y_pred, y_true, alpha=0.8, gamma=2.0):
    """血管分割Focal Loss - 处理类别不平衡

    由来: Lin et al. 2017 Focal Loss for Dense Object Detection
    目的: 解决血管分割中前景背景极端不平衡问题
    适用场景: 细血管分割、小目标检测
    医学影像搭配: 视网膜微血管、毛细血管分割
    """
    bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    pt = torch.exp(-bce_loss)  # pt = p if y=1, 1-p if y=0

    # 根据标签选择alpha
    alpha_factor = torch.where(y_true == 1, alpha, 1 - alpha)
    focal_loss = alpha_factor * (1 - pt) ** gamma * bce_loss

    return focal_loss.mean()


def iou_loss(y_pred, y_true, smooth=1e-6):
    """IoU损失 - 直接优化Jaccard相似系数

    由来: Jaccard相似系数的直接优化版本
    目的: 提供与评估指标更一致的学习信号
    适用场景: 需要与IoU指标强相关的任务
    医学影像搭配: 各种医学影像分割任务

    参数:
        y_pred: 模型输出 [B, C, H, W] (应为sigmoid后)
        y_true: 真实标签 [B, C, H, W]
    """
    y_pred = y_pred.contiguous().view(y_pred.shape[0], -1)
    y_true = y_true.contiguous().view(y_true.shape[0], -1)

    intersection = (y_pred * y_true).sum(dim=1)
    union = y_pred.sum(dim=1) + y_true.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou.mean()


# 损失函数注册表
LOSS_REGISTRY = {
    'iou': iou_loss,
    'dice': dice_loss,
    'cl_dice': cl_dice_loss,
    'bce_dice': bce_dice_loss,
    'focal': vascular_focal_loss,
    'combined_cl_dice': soft_cl_dice_combined_loss,
}
