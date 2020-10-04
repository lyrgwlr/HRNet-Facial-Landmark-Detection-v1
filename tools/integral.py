import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super(SoftmaxCrossEntropyWithLogits, self).__init__()

    def forward(self, input, target):
        loss = torch.sum(-target * F.log_softmax(input, -1), -1)
  
      
        mean_loss = torch.mean(loss)
        return mean_loss


class MixedLoss(nn.Module):
    '''
    ref: https://github.com/mks0601/PoseFix_RELEASE/blob/master/main/model.py
    input: {
        'heatmap': (N, C, h,w) unnormalized
        'coord': (N, C, 3)
    }
    target: {
        'heatmap': (N, C, 4,h*w), normalized
        'coord': (N, C, 3)
    }
    '''
    def __init__(self, heatmap_weight=0.5):
    
        super(MixedLoss, self).__init__()
        self.w1 = heatmap_weight
        self.w2 = 1 - self.w1
        self.cross_entropy_loss = SoftmaxCrossEntropyWithLogits()

    def forward(self, pred_heatmap, target, gt_coord):
        
        #gt_heatmap, gt_coord = target['heatmap'], target['coord']
        normalized_heatmap, pred_coord = soft_argmax(pred_heatmap, pred_heatmap.shape[1])
        # Heatmap loss
        N, C = normalized_heatmap.shape[0:2]
        normalized_heatmap = normalized_heatmap.view(N*C, -1)
        # pred_heatmap = pred_heatmap.view(N*C, -1)
        
        gt_heatmap = target.view(N*C,  -1)

        assert normalized_heatmap.shape == gt_heatmap.shape
        # assert pred_heatmap.shape == gt_heatmap.shape
        _assert_no_grad(gt_heatmap)
        _assert_no_grad(gt_coord)
       
        hm_loss = self.cross_entropy_loss(normalized_heatmap, gt_heatmap)

        # Coord L1 loss
        l1_loss = torch.mean(torch.abs(pred_coord - gt_coord))

        return self.w1 * hm_loss, self.w2 * l1_loss, pred_coord

class MixedLoss_v2(nn.Module):
    '''
    ref: https://github.com/mks0601/PoseFix_RELEASE/blob/master/main/model.py
    input: {
        'heatmap': (N, C, h,w) unnormalized
        'coord': (N, C, 3)
    }
    target: {
        'heatmap': (N, C, 4,h*w), normalized
        'coord': (N, C, 3)
    }
    '''
    def __init__(self, heatmap_weight=0.5):
    
        super(MixedLoss_v2, self).__init__()
        self.w1 = heatmap_weight
        self.w2 = 1 - self.w1
        self.mse_loss = torch.nn.MSELoss(size_average=True).cuda()

    def forward(self, pred_heatmap, target, gt_coord):
        
        _, pred_coord = soft_argmax(pred_heatmap, pred_heatmap.shape[1])
        
        assert pred_heatmap.shape == target.shape
        _assert_no_grad(target)
        _assert_no_grad(gt_coord)
       
        hm_loss = self.mse_loss(torch.sigmoid(pred_heatmap), target)

        # Coord L1 loss
        l1_loss = torch.mean(torch.abs(pred_coord - gt_coord))

        return self.w1 * hm_loss , self.w2 * l1_loss, pred_coord

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

def soft_argmax(heatmaps, joint_num):
    assert isinstance(heatmaps, torch.Tensor)
    h ,w = heatmaps.shape[-2], heatmaps.shape[-1]
    heatmaps = heatmaps.reshape((-1, joint_num, h*w))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, h, w))

    accu_x = heatmaps.sum(dim=2)
    accu_y = heatmaps.sum(dim=3)
    

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(1, w+1).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(1, h+1).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    

    accu_x = accu_x.sum(dim=2, keepdim=True) -1
    accu_y = accu_y.sum(dim=2, keepdim=True) -1
    

    coord_out = torch.cat((accu_x, accu_y), dim=2) #(B,c,2)

    return heatmaps, coord_out

