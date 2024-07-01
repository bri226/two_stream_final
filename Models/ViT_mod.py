import torch
from torch import nn 

# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels:int=3,patch_size:int=16,embedding_dim:int=768):
        super().__init__()
        
        self.patch_size = patch_size
        
        self.patcher = nn.Conv2d(in_channels=in_channels,out_channels=embedding_dim,
                                 kernel_size=patch_size,stride=patch_size,padding=0)
        
        self.flatten = nn.Flatten(start_dim=2,end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
        return x_flattened.permute(0, 2, 1)

class ViT(nn.Module): 
  def __init__(self,
               img_size=224,
               num_channels=3,
               patch_size=16,
               embedding_dim=768,
               dropout=0.1, 
               mlp_size=3072,
               num_transformer_layers=12,
               num_heads=12,
               num_classes=6,
               freeze_layers_until=None):
    super().__init__()

    assert img_size % patch_size == 0, "Image size must be divisble by patch size."

    self.patch_embedding = PatchEmbedding(in_channels=num_channels,patch_size=patch_size,embedding_dim=embedding_dim)

    self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),requires_grad=True)

    num_patches = (img_size * img_size) // patch_size**2
    self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim))

    self.embedding_dropout = nn.Dropout(p=dropout)

    # # 5. Create Transformer Encoder layer (single)
    # self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
    #                                                             nhead=num_heads,
    #                                                             dim_feedforward=mlp_size,
    #                                                             activation="gelu",
    #                                                             batch_first=True,
    #                                                             norm_first=True)

    # 5. Create stack Transformer Encoder layers (stacked single layers)
    self.transformer_encoder = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                        nhead=num_heads,
                        dim_feedforward=mlp_size,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True), # Create a single Transformer Encoder Layer
                        num_layers=num_transformer_layers) # Stack it N times

    # # 7. Create MLP head
    # self.mlp_head = nn.Sequential(
    #     nn.LayerNorm(normalized_shape=embedding_dim),
    #     nn.Linear(in_features=embedding_dim,
    #               out_features=num_classes)
    # )

    # Freeze specified layers
    self.freeze_layers(freeze_layers_until)

  def freeze_layers(self, layer_idx):
    for param in self.patch_embedding.parameters():
        param.requires_grad = False
    self.embedding_dropout.eval()
    for param in self.embedding_dropout.parameters():
        param.requires_grad = False

    for i in range(layer_idx):
        for param in self.transformer_encoder.layers[i].parameters():
            param.requires_grad = False

  def forward(self, x):
    batch_size = x.shape[0]
    x = self.patch_embedding(x)
    class_token = self.class_token.expand(batch_size, -1, -1) 
    x = torch.cat((class_token, x), dim=1)
    x = self.positional_embedding + x
    x = self.embedding_dropout(x)
    x = self.transformer_encoder(x)
    # x = self.mlp_head(x[:, 0])

    return x
