import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_u = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.W_o = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x):
        X_u = torch.sigmoid(self.W_u(x))
        X_o = torch.tanh(self.W_o(x))
        return X_u * X_o


class STGNNDetector(nn.Module):
    def __init__(self, P=4, N=256):
        super().__init__()
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
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_val = 0.1
        self.alpha = 0.2

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a1 = nn.Parameter(torch.zeros(out_features, 1))
        self.a2 = nn.Parameter(torch.zeros(out_features, 1))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        B, C, N = input.size()

        x = input.permute(0, 2, 1).reshape(-1, self.in_features)
        Wh = torch.matmul(x, self.W)
        Wh = Wh.view(B, N, self.out_features)

        e1 = torch.matmul(Wh, self.a1).squeeze(-1)
        e2 = torch.matmul(Wh, self.a2).squeeze(-1)
        e = e1.unsqueeze(2) + e2.unsqueeze(1)
        e = self.leakyrelu(e)

        zero_vec = -9e15 * torch.ones_like(e)
        if adj.dim() == 3:
            adj_2d = adj[0]
        else:
            adj_2d = adj
        attention = torch.where(adj_2d > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout_val, training=self.training)

        h_prime = torch.matmul(attention, Wh)
        return h_prime.permute(0, 2, 1)


class STFE(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()
        self.gat = GraphAttentionLayer(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x_list):
        N = x_list[0].size(2)
        P = len(x_list)
        adj = self._generate_graph(x_list)

        outputs = []
        for p in range(P):
            h = self.gat(x_list[p], adj)
            h = self.relu(h)
            outputs.append(h)

        return torch.stack(outputs, dim=2)

    def _generate_graph(self, x_list):
        N = x_list[0].size(2)
        adj = torch.eye(N, device=x_list[0].device)
        adj = adj + torch.diag(torch.ones(N - 1, device=x_list[0].device), 1)
        adj = adj + torch.diag(torch.ones(N - 1, device=x_list[0].device), -1)
        return adj


class Detector(nn.Module):
    def __init__(self, in_features=1024, hidden_features=512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_features, 2, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)

    def predict_sample_level(self, logits):
        per_range_bin_probs = torch.softmax(logits, dim=1)
        target_probs = per_range_bin_probs[:, 1, :]
        max_target_prob, _ = torch.max(target_probs, dim=1)
        sample_level_probs = torch.stack([1 - max_target_prob, max_target_prob], dim=1)
        return sample_level_probs, per_range_bin_probs


class FARController(nn.Module):
    def __init__(self):
        super().__init__()

    def calculate_threshold(self, clutter_probabilities, target_pfa):
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
        return o[threshold_idx]

    def default_threshold(self, target_pfa):
        return target_pfa

    def calibrate(self, clutter_scores, target_pfa=0.001):
        if clutter_scores.ndim == 3:
            clutter_probs = clutter_scores[:, 0, :].flatten()
        elif clutter_scores.ndim == 2:
            clutter_probs = clutter_scores[:, 0]
        else:
            clutter_probs = clutter_scores

        return self.calculate_threshold(clutter_probs, target_pfa)

    def decide(self, detection_output, threshold):
        clutter_probs = detection_output[:, 0, :]
        return (clutter_probs <= threshold).float()
