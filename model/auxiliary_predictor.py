import torch
import torch.nn as nn

class MLP(nn.Module):
    '''An auxiliary classifier to be trained jointly the VQ-VAE'''
    
    def __init__(self, latent_size, num_classes):
        super(MLP, self).__init__()
        self.latent_size = latent_size
        
        # MLP Classifier
        self.mlp = nn.Sequential(
            nn.Linear(latent_size , 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, z, targets):
        class_logits = self.mlp(z)
        loss = self.criterion(input=class_logits, target=targets)
        avg_loss = torch.mean(loss)

        return avg_loss, class_logits

