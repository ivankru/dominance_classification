import torch
import torch.nn.functional as F
from typing import Any


class NormalizedCrossEntropy(torch.nn.Module):
    # normalized cross-entropy
    def __init__(self, num_classes: int, scale: float = 1.0, weight: Any = None, device: str = 'cpu') -> None:
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        self.weight = weight

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        
        positions = torch.argmax(label_one_hot, dim = 1)
        weight = self.weight[positions]
        weight = torch.unsqueeze(weight, dim=1)
        weight = weight.repeat(1, self.num_classes)
        
        nce = -1 * torch.sum(weight * label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()
    
class NormalizedCrossEntropy_normal(torch.nn.Module):
    # normalized cross-entropy
    def __init__(self, num_classes: int, scale: float = 1.0, weight: Any = None, device: str = 'cpu') -> None:
        super(NormalizedCrossEntropy_normal, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        # self.weight = torch.tensor(weight).float()
        self.weight = weight

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        
        positions = torch.argmax(label_one_hot, dim = 1)

        weight = self.weight[positions]
        weight = torch.unsqueeze(weight, dim=1)
        weight = weight.repeat(1, self.num_classes)
                
        weight_sum = weight[:, 0].sum()
        
        nce = -1 * torch.sum(weight * label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        nce = nce.sum() / weight_sum
        
        return self.scale * nce

class CustomCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes: int, scale: float = 1.0, weight: Any = None, device: str = 'cpu') -> None:
        super(CustomCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        self.weight = weight

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        
        positions = torch.argmax(label_one_hot, dim = 1)
        weight = self.weight[positions]
        weight = torch.unsqueeze(weight, dim=1)
        weight = weight.repeat(1, self.num_classes)
        
        ce = -1 * torch.sum(weight * label_one_hot * pred, dim=1)
        return self.scale * ce.mean()

class CustomCrossEntropy_normal(torch.nn.Module):
    def __init__(self, num_classes: int, scale: float = 1.0, weight: Any = None, device: str = 'cpu') -> None:
        super(CustomCrossEntropy_normal, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        self.weight = weight

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        
        positions = torch.argmax(label_one_hot, dim = 1)
        
        weight = self.weight[positions]
        weight = torch.unsqueeze(weight, dim=1)
        weight = weight.repeat(1, self.num_classes)
                
        weight_sum = weight[:, 0].sum()
                    
        ce = -1 * torch.sum(weight * label_one_hot * pred, dim=1)
        ce = ce.sum() / weight_sum
        
        return ce * self.scale
    
class ReverseCrossEntropy(torch.nn.Module):
    # reverse cross-entropy
    def __init__(self, num_classes: int, scale: float = 1.0, weight: Any = None, device: str = 'cpu') -> None:
        super(ReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        self.weight = weight

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        
        positions = torch.argmax(label_one_hot, dim = 1)
        weight = self.weight[positions]
        weight = torch.unsqueeze(weight, dim=1)
        weight = weight.repeat(1, self.num_classes)
        
        rce = (-1*torch.sum(weight * pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()
    
class ReverseCrossEntropy_normal(torch.nn.Module):
    # reverse cross-entropy
    def __init__(self, num_classes: int, scale: float = 1.0, weight: Any = None, device: str = 'cpu') -> None:
        super(ReverseCrossEntropy_normal, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        self.weight = weight

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        
        positions = torch.argmax(label_one_hot, dim = 1)
        
        weight = self.weight[positions]
        weight = torch.unsqueeze(weight, dim=1)
        weight = weight.repeat(1, self.num_classes)
        
        weight_sum = weight[:, 0].sum()
        
        rce = (-1*torch.sum(weight * pred * torch.log(label_one_hot), dim=1))
        rce = rce.sum() / weight_sum
        
        return self.scale * rce

class NCEandRCE(torch.nn.Module):
    # normalized cross-entropy + reverse cross-entropy
    def __init__(self, alpha: float, beta: float, num_classes: int, weight: Any = None,
                 device: str = 'cpu', **kwargs) -> None:
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        weight = torch.tensor(weight).type(torch.double).to(device)
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes, weight=weight, device=device)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes, weight=weight, device=device)

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.nce(pred, labels) + self.rce(pred, labels)
    
class NCEandRCE_normal(torch.nn.Module):
    # normalized cross-entropy + reverse cross-entropy
    def __init__(self, alpha: float, beta: float, num_classes: int, weight: Any = None,
                 device: str = 'cpu', **kwargs) -> None:
        super(NCEandRCE_normal, self).__init__()
        self.num_classes = num_classes
        weight = torch.tensor(weight).type(torch.double).to(device)
        self.nce = NormalizedCrossEntropy_normal(scale=alpha, num_classes=num_classes, weight=weight, device=device)
        self.rce = ReverseCrossEntropy_normal(scale=beta, num_classes=num_classes, weight=weight, device=device)

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.nce(pred, labels) + self.rce(pred, labels)

class CEandRCE(torch.nn.Module):
    # cross-entropy + reverse cross-entropy
    def __init__(self, alpha: float, beta: float, num_classes: int, weight: Any = None,
                 device: str = 'cpu', **kwargs) -> None:
        super(CEandRCE, self).__init__()
        self.num_classes = num_classes
        weight = torch.tensor(weight).type(torch.double).to(device)
        self.ce = CustomCrossEntropy(scale=alpha, num_classes=num_classes, weight=weight, device=device)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes, weight=weight, device=device)

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(pred, labels) + self.rce(pred, labels)

class CEandRCE_normal(torch.nn.Module):
    # cross-entropy + reverse cross-entropy
    def __init__(self, alpha: float, beta: float, num_classes: int, weight: Any = None,
                 device: str = 'cpu', **kwargs) -> None:
        super(CEandRCE_normal, self).__init__()
        self.num_classes = num_classes
        weight = torch.tensor(weight).type(torch.double).to(device)
        self.ce = CustomCrossEntropy_normal(scale=alpha, num_classes=num_classes, weight=weight, device=device)
        self.rce = ReverseCrossEntropy_normal(scale=beta, num_classes=num_classes, weight=weight, device=device)

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(pred, labels) + self.rce(pred, labels)

    
class CE(torch.nn.Module):
    # cross-entropy 
    def __init__(self, num_classes: int, weight: Any = None, device: str = 'cpu', **kwargs) -> None:
        super(CE, self).__init__()
        self.num_classes = num_classes
        weight = torch.tensor(weight).type(torch.double).to(device)
        self.ce = CustomCrossEntropy(scale=1, num_classes=num_classes, weight=weight, device=device)

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(pred, labels)

class CE_normal(torch.nn.Module):
    # cross-entropy 
    def __init__(self, num_classes, weight=None, device='cpu', **kwargs) -> None:
        super(CE_normal, self).__init__()
        self.num_classes = num_classes
        weight = torch.tensor(weight).type(torch.double).to(device)
        self.ce = CustomCrossEntropy_normal(scale=1, num_classes=num_classes, weight=weight, device=device)

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(pred, labels)
