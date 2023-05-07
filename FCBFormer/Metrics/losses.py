import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        print(targets)
        num = targets.size(0) # batch size

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2 # aka TP
        
        fp = (1 - m2) * m1 # (B, 352*352)
        tn = (1 - m1) * (1 - m2) # (B, 352*352)
        # n_tn = torch.count_nonzero(tn, dim=1) # count real TN, i.e TN without 0s. Shape: (B,1)
        
        sorted_tn, indices = torch.sort(tn, dim=1, descending=True)
        K = 1 # 1 means all sorted_tn is contained in one bin
        M = int(0.1 * tn.size(1))
        extra = sorted_tn[:, :M]
        
        # FDLv2
        # B = torch.cat((fp, extra), dim=1) # (B, 352^2 + extra)
        # score = (
        #     (intersection.sum(1) + self.smooth)
        #     / B.sum(1) + m2.sum(1) + self.smooth)
        # )
        
        # FDLv1
        fn = m2 * (1 - m1)
        AB = torch.cat((intersection, fn, fp, extra), dim=1) # (B, 352^2 + ...), 0 included
        score = (
            2 * (intersection.sum(1) + self.smooth)
            / (AB.sum(1) + m2.sum(1) + self.smooth)
        )
        
        score = 1 - score.sum() / num
        return score


    