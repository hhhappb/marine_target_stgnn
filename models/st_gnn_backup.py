import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TFE, self).__init__()
        self.W_u = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.W_o = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x):
        X_u = torch.sigmoid(self.W_u(x))
        X_o = torch.tanh(self.W_o(x))
        X_t = X_u * X_o
        return X_t


class STGNNDetector(nn.Module):
    def __init__(self, P=4, N=256):
        super(STGNNDetector, self).__init__()
        self.P = P
        self.N = N

        self.ft1 = nn.Conv2d(2, 32, kernel_size=(1, 3), padding=(0, 1))
        self.ft2 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        self.ft_relu = nn.ReLU()

        self.sfe1 = STFE(in_channels=64, out_channels=128)
        self.tfe1 = TFE(in_channels=128, out_channels=256)
        self.sfe2 = STFE(in_channels=256, out_channels=512)
        self.tfe2 = TFE(in_channels=512, out_channels=1024)

        self.detector = Detector(1024, 512)
        
        self.far_controller = FARController()

    def forward(self, E, return_features=False):
        E_real = E.real
        E_imag = E.imag
        x = torch.stack([E_real, E_imag], dim=1)

        x = self.ft_relu(self.ft1(x))
        x = self.ft_relu(self.ft2(x))

        F_features = x

        x_list = [F_features[:, :, p, :] for p in range(self.P)]

        sfe1_out = self.sfe1(x_list)
        tfe1_out = self.tfe1(sfe1_out)
        
        tfe1_list = [tfe1_out[:, :, p, :] for p in range(tfe1_out.size(2))]
        sfe2_out = self.sfe2(tfe1_list)
        tfe2_out = self.tfe2(sfe2_out)

        temporal_features = tfe2_out.squeeze(2)

        per_range_bin_logits = self.detector(temporal_features)
        sample_level_probs, _ = self.detector.predict_sample_level(per_range_bin_logits)

        if return_features:
            return sample_level_probs, per_range_bin_logits, F_features, tfe2_out
        return sample_level_probs
    
    def predict_with_far_control(self, E, target_pfa=0.001, calibration_data=None):
        """
        带虚警率控制的预测，仅在测试阶段使用
        
        :param E: 输入数据，形状 [B, P, N]
        :param target_pfa: 目标虚警率 α_f
        :param calibration_data: 用于校准的纯杂波样本（可选），形状 [B_c, P, N]
        :return: 预测结果（二值）、检测输出、阈值
        """
        self.eval()
        with torch.no_grad():
            _, per_range_bin_outputs, _, _ = self.forward(E, return_features=True)
            probs = torch.softmax(per_range_bin_outputs, dim=1)
            
            if calibration_data is not None:
                _, calib_outputs, _, _ = self.forward(calibration_data, return_features=True)
                calib_probs = torch.softmax(calib_outputs, dim=1)
                threshold = self.far_controller.calibrate(calib_probs.cpu().numpy(), target_pfa)
            else:
                threshold = self.far_controller.default_threshold(target_pfa)
            
            predictions = self.far_controller.decide(probs, threshold)
            
            return predictions, probs, threshold


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, input, adj):
        B, C, N = input.size()
        h = torch.matmul(input.transpose(1, 2), self.W)

        a_input = torch.cat([h.unsqueeze(2).expand(-1, -1, N, -1),
                            h.unsqueeze(2).transpose(1, 2).expand(-1, N, -1, -1)], dim=-1)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, 0.3, training=self.training)

        h_prime = torch.matmul(attention, h)
        return h_prime.transpose(1, 2)


class STFE(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super(STFE, self).__init__()
        self.gat = GraphAttentionLayer(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x_list):
        B = x_list[0].size(0)
        N = x_list[0].size(2)
        P = len(x_list)

        adj = self._generate_graph(x_list)

        outputs = []
        for p in range(P):
            h = self.gat(x_list[p], adj)
            h = self.relu(h)
            outputs.append(h)

        out = torch.stack(outputs, dim=2)

        return out

    def _generate_graph(self, x_list):
        B = x_list[0].size(0)
        N = x_list[0].size(2)
        P = len(x_list)

        adj = torch.eye(N, device=x_list[0].device)
        adj = adj + torch.diag(torch.ones(N - 1, device=x_list[0].device), 1)
        adj = adj + torch.diag(torch.ones(N - 1, device=x_list[0].device), -1)
        adj = adj.unsqueeze(0).repeat(B, 1, 1)
        return adj


class Detector(nn.Module):
    def __init__(self, in_features=1024, hidden_features=512):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_features, 2, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

    def predict_sample_level(self, logits):
        """直接从logits计算样本级别的概率，避免重复前向传播"""
        per_range_bin_probs = torch.softmax(logits, dim=1)
        target_probs = per_range_bin_probs[:, 1, :]
        max_target_prob, _ = torch.max(target_probs, dim=1)
        sample_level_probs = torch.stack([1 - max_target_prob, max_target_prob], dim=1)
        return sample_level_probs, per_range_bin_probs


class FARController(nn.Module):
    """
    虚警率控制器 (FAR Controller)
    根据期望虚警率设置检测阈值，仅测试阶段使用，不参与训练
    
    论文算法：
    1. 取训练集中所有杂波样本的 o(0)（杂波概率）
    2. 升序排列：o = [o(0)_1, o(0)_2, ..., o(0)_{N_c}]
    3. 根据期望虚警率 α_f 计算阈值：h = o(⌈α_f * N_c⌉)
    4. 测试时判决：o(0) > h → 杂波；o(0) ≤ h → 目标
    """
    
    def __init__(self):
        super(FARController, self).__init__()
    
    def calculate_threshold(self, clutter_probabilities, target_pfa):
        """
        根据杂波样本的杂波概率计算检测阈值
        
        :param clutter_probabilities: 杂波样本的杂波概率 o(0)，形状 [N_c]
        :param target_pfa: 目标虚警率 α_f
        :return: 检测阈值 h
        """
        if isinstance(clutter_probabilities, torch.Tensor):
            o = clutter_probabilities.cpu().numpy()
        else:
            o = np.array(clutter_probabilities)
        
        o = np.sort(o)
        N_c = len(o)
        
        if N_c == 0:
            return self.default_threshold(target_pfa)
        
        threshold_idx = int(np.ceil(target_pfa * N_c)) - 1
        threshold_idx = max(0, min(threshold_idx, N_c - 1))
        h = o[threshold_idx]
        
        return h
    
    def default_threshold(self, target_pfa):
        """
        使用默认方法计算阈值（当没有校准数据时）
        
        :param target_pfa: 目标虚警率
        :return: 检测阈值
        """
        return target_pfa
    
    def calibrate(self, clutter_scores, target_pfa=0.001):
        """
        校准方法：使用纯杂波数据计算阈值
        
        :param clutter_scores: 纯杂波样本的预测结果（形状 [B, 2, N] 或 [B, 2]）
        :param target_pfa: 目标虚警率
        :return: 检测阈值
        """
        if clutter_scores.ndim == 3:
            clutter_probs = clutter_scores[:, 0, :].flatten()
        elif clutter_scores.ndim == 2:
            clutter_probs = clutter_scores[:, 0]
        else:
            clutter_probs = clutter_scores
        
        return self.calculate_threshold(clutter_probs, target_pfa)
    
    def decide(self, detection_output, threshold):
        """
        根据检测输出和阈值进行判决
        
        :param detection_output: Ds的输出，形状 [B, 2, N]，其中第0通道是杂波概率 o(0)
        :param threshold: 检测阈值 h
        :return: 判决结果，形状 [B, N]，0=杂波，1=目标
        """
        clutter_probs = detection_output[:, 0, :]
        decisions = (clutter_probs <= threshold).float()
        return decisions