"""损失函数定义 - 专注训练优化"""
import torch
import torch.nn.functional as F


def dice_loss(y_pred, y_true, smooth=1.):
    """经典Dice损失 - 用于分割任务
    参数:
        y_pred: 模型输出 [B, C, H, W]
        y_true: 真实标签 [B, C, H, W]
        smooth: 平滑项避免除零
    返回:
        dice_loss: 标量损失值
    """
    y_pred = y_pred.contiguous()
    y_true = y_true.contiguous()
    intersection = (y_pred * y_true).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) /
                (y_pred.sum(dim=2).sum(dim=2) + y_true.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()


def bce_dice_loss(y_pred, y_true):
    """BCE + Dice组合损失 - 平衡边界和区域优化"""
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
    dice = dice_loss(torch.sigmoid(y_pred), y_true)
    return bce + dice


def cl_dice_loss(pred, target, smooth=1e-6):
    """经典clDice损失 - 血管分割专用
    参考: https://arxiv.org/abs/2003.07311
    实现中心线感知的Dice损失，专注血管结构的拓扑保持
    """
    # 提取中心线 - 使用可微的形态学操作
    pred_center = _extract_centerline(torch.sigmoid(pred))
    target_center = _extract_centerline(target)

    # 计算clDice的两个分量
    tprec = (_soft_cldice(pred_center, target, smooth) +
             _soft_cldice(target_center, torch.sigmoid(pred), smooth)) / 2

    return 1 - tprec


def _extract_centerline(tensor):
    """可微的中心线提取 - 使用迭代腐蚀膨胀"""
    # 简化的可微中心线近似
    # 实际应用中可以使用更复杂的可微分形态学操作
    eroded = F.max_pool2d(1 - tensor, kernel_size=3, stride=1, padding=1)
    dilated = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    centerline = torch.abs(tensor - dilated * (1 - eroded))
    return centerline


def _soft_cldice(center, surface, smooth=1e-6):
    """软clDice计算 - 支持可微操作"""
    intersection = (center * surface).sum()
    return (2. * intersection + smooth) / (center.sum() + surface.sum() + smooth)


def vascular_focal_loss(y_pred, y_true, alpha=0.8, gamma=2.0):
    """血管分割Focal Loss - 处理类别不平衡"""
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    pt = torch.exp(-bce)
    focal_loss = alpha * (1 - pt) ** gamma * bce
    return focal_loss.mean()


# 损失函数注册表 - 简洁的访问方式
LOSS_REGISTRY = {
    'dice': dice_loss,
    'bce_dice': bce_dice_loss,
    'cl_dice': cl_dice_loss,
    'focal': vascular_focal_loss
}
