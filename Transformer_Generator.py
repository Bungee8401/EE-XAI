import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CustomDataset import Data_prep_224_gen
import pytorch_lightning as pl

# Hyperparameters
image_size = 224
num_classes = 10
batch_size = 64
learning_rate = 1e-4
num_epochs = 10
d_model = 512  # Embedding size
num_heads = 8  # Number of attention heads
num_layers = 6  # Number of transformer layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pl.seed_everything(2024)

root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
dataprep = Data_prep_224_gen(root)
train_idx, val_idx, test_idx = dataprep.get_category_index(category = 0) # 0 airplane, 3 cat, 8 ship
# print(f"Total entries in train_idx: {len(train_idx)}, val_idx: {len(val_idx)}, test_idx: {len(test_idx)}")
trainloader, valloader, testloader = dataprep.create_catogery_loaders(batch_size=256, num_workers=8, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)



# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

# Generator
class Generator(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, image_size):
        super(Generator, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, image_size * image_size, d_model))
        self.embedding = nn.Linear(3 * image_size * image_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, 3 * image_size * image_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x.view(-1, 3, image_size, image_size)

# Training
def train(generator, data_loader, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

    generator.train()
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            images = images.view(images.size(0), -1)
            generated_images = generator(images)
            loss = criterion(generated_images, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')


# Initialize and train
generator = Generator(d_model, num_heads, num_layers, image_size)
train(generator, trainloader, 50, 0.001)



































































# import torch
# import pytorch_lightning as pl
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import os
# import torch.optim as optim
# import Alex_ee_inference
# from Alexnet_early_exit import BranchedAlexNet
# import matplotlib.pyplot as plt
# from CustomDataset import Data_prep_224_gen
# import datetime
# from tqdm import tqdm
# import pickle
# import random
# from torch.utils.data import DataLoader, TensorDataset
# from einops import rearrange, repeat
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, dim, heads=8, dropout=0.1):
#         super().__init__()
#         assert dim % heads == 0, "dim必须能被heads整除"
#         self.heads = heads
#         self.scale = (dim // heads) ** -0.5  # 缩放因子
#
#         # 线性变换：Q/K/V生成
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         # 输出投影
#         self.to_out = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x, context=None, mask=None):
#         """
#         输入：
#         x:      [batch, seq_len, dim]  (默认自注意力)
#         context: [batch, seq_len2, dim] (交叉注意力时提供)
#         mask:   [batch, seq_len, seq_len2]
#         """
#         # 如果没有外部context，则为自注意力
#         context = x if context is None else context
#
#         # 生成Q/K/V
#         qkv = self.to_qkv(x if context is x else torch.cat([x, context], dim=1))
#         q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d',
#                             qkv=3, h=self.heads)  # [b, h, n, d]
#
#         # 注意力计算
#         attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#         if mask is not None:
#             attn = attn.masked_fill(mask == 0, -1e9)
#         attn = attn.softmax(dim=-1)
#
#         # 加权聚合Value
#         out = torch.einsum('bhij,bhjd->bhid', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#
#         return self.to_out(out)
#
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, heads, mlp_dim, dropout):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = MultiHeadAttention(dim, heads)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         # 自注意力
#         x = x + self.attn(self.norm1(x))
#         # MLP
#         x = x + self.mlp(self.norm2(x))
#         return x
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, heads=4):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.attn = MultiHeadAttention(dim, heads)
#
#     def forward(self, x, cls_feats):
#         """
#         x: 生成器特征 [B, N+1, dim]
#         cls_feats: 分类器中间层特征 [B, M, dim]
#         """
#         x = self.norm(x)
#         # 用分类器特征作为Key/Value，生成器特征作为Query
#         return x + self.attn(x, cls_feats, cls_feats)
#
# class UnfoldReconstructor(nn.Module):
#     """将分块特征重建为完整图像"""
#
#     def __init__(self, patch_size, image_size):
#         super().__init__()
#         self.patch_size = patch_size
#         self.image_size = image_size
#
#     def forward(self, x):
#         # x: [B, patch_dim, num_patches]
#         x = rearrange(x, 'b (c p1 p2) (h w) -> b c (h p1) (w p2)',
#                       p1=self.patch_size, p2=self.patch_size,
#                       h=self.image_size // self.patch_size)
#         return x
#
# class ViTGenerator(nn.Module):
#     def __init__(self,
#                  image_size=224,
#                  patch_size=4,
#                  in_channels=3,
#                  dim=128,
#                  depth=6,
#                  heads=8,
#                  mlp_dim=256,
#                  dropout=0.1):
#         super().__init__()
#
#         # 图像分块嵌入
#         self.patch_size = patch_size
#         num_patches = (image_size // patch_size) ** 2
#         patch_dim = in_channels * patch_size ** 2
#
#         self.patch_embed = nn.Sequential(
#             nn.Unfold(kernel_size=patch_size, stride=patch_size),  # [B, C*p*p, N]
#             nn.Linear(patch_dim, dim)  # [B, N, dim]
#         )
#
#         # 位置编码
#         self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#
#         # Transformer编码器
#         self.encoder = nn.ModuleList([
#             TransformerBlock(dim, heads, mlp_dim, dropout)
#             for _ in range(depth)
#         ])
#
#         # 交叉注意力模块（关注分类器特征）
#         self.cross_attn = CrossAttentionBlock(dim)
#
#         # 图像重建解码器
#         self.decoder = nn.Sequential(
#             nn.Linear(dim, patch_dim),
#             nn.UnfoldReconstructor(patch_size, image_size)  # 自定义重建层
#         )
#
#     def forward(self, x, cls_feats=None):
#         # 输入分块嵌入 [B, C, H, W] → [B, N, dim]
#         x = self.patch_embed(x).transpose(1, 2)  # [B, N, dim]
#         b, n, _ = x.shape
#
#         # 添加CLS Token和位置编码
#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
#         x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, dim]
#         x += self.pos_embed
#
#         # Transformer编码
#         for blk in self.encoder:
#             x = blk(x)
#
#         # 交叉注意力（与分类器特征交互）
#         if cls_feats is not None:
#             x = self.cross_attn(x, cls_feats)  # 利用分类器特征引导生成
#
#         # 解码重建图像
#         x = x[:, 1:]  # 移除CLS Token
#         x = self.decoder(x.transpose(1, 2))  # [B, C, H, W]
#         return x
#
# def compute_loss(gen_img, real_img):
#     # 空间加权重构损失（强化中心区域）
#     recon_loss = nn.L1Loss()(gen_img, real_img)
#
#     # 分类器特征对齐损失（迫使生成器欺骗分类器早期层）
#     gen_feature = B_alex.extract_features(gen_img)
#     ori_feature = B_alex.extract_features(real_img)
#     feat_loss = nn.MSELoss()(gen_feature, ori_feature)
#
#     return recon_loss + 0.5 * feat_loss
#
# def crop_img(imgs):
#     if crop_size != 0 and replace_size == 0:  # crop
#         cropped_imgs = imgs[:, :, crop_size:-crop_size, crop_size:-crop_size]  # [B, C, 184, 184]
#         final_imgs = torch.nn.functional.interpolate(cropped_imgs, size=(224, 224), mode="bilinear",
#                                                      align_corners=False)
#     elif crop_size == 0 and replace_size != 0:  # replace
#         processed_imgs = imgs.clone()
#         processed_imgs[:, :, :replace_size, :] = 1  # Replace top edge
#         processed_imgs[:, :, -replace_size:, :] = 1  # Replace bottom edge
#         processed_imgs[:, :, :, :replace_size] = 1  # Replace left edge
#         processed_imgs[:, :, :, -replace_size:] = 1  # Replace right edge
#         final_imgs = processed_imgs
#     # elif crop_size == 0 and replace_size == 0 and only_edge != 0:
#     #     processed_imgs = imgs.clone()
#     #     processed_imgs[:, :, only_edge:-only_edge, only_edge:-only_edge] = 0
#     #     final_imgs =  processed_imgs
#
#     else:
#         final_imgs = imgs
#
#     return final_imgs
#
# def train(num_epoch):
#     generator.train()
#     loss_log_path = r"/home/yibo/PycharmProjects/Thesis/training_weights/Transformer/log.txt"
#     epoch_loss = 0  # To accumulate loss for the entire epoch
#     num_batches = 0 # To calculate average loss by dividing by # of batches
#
#     for epoch in range(num_epoch):
#         for idx, data in enumerate(trainloader):
#             original_images, original_labels = data[0].to(device), data[1].to(device)
#
#             optimizer.zero_grad()
#
#             generated_images = generator(original_images)
#             loss = compute_loss(B_alex, generated_images, original_images, original_labels)
#
#             loss.backward()
#             optimizer.step()
#
#             # Accumulate the loss for the current batch
#             epoch_loss += loss.item()
#             num_batches += 1
#
#         avg_loss = epoch_loss / num_batches
#         print(f"Epoch {epoch + 1}/{num_epoch}, Training Loss: {avg_loss}")
#
#         # Append the average loss to the local file
#         with open(loss_log_path, "a") as f:
#             current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             f.write(f"[{current_time}] Epoch {epoch + 1}: {avg_loss}\n")
#
#         if not os.path.exists(r"/home/yibo/PycharmProjects/Thesis/training_weights/Transformer"):
#                 os.makedirs(r"/home/yibo/PycharmProjects/Thesis/training_weights/Transformer")
#         if (epoch+1) % 50 == 0:
#             torch.save(generator.state_dict(),
#                     f"/home/yibo/PycharmProjects/Thesis/training_weights/Transformer/Generator_epoch_{epoch+1}.pth")
#
# def test(batch, num):
#     generator.eval()
#     B_alex.eval()
#     num_forced_class_original = 0
#     num_forced_class_generated = 0
#
#     with torch.no_grad():
#         for idx, data in enumerate(testloader):
#             print(f"batch {idx+1}")
#             if idx >= batch:
#                 break
#
#             original_images, original_labels = data[0].to(device), data[1].to(device)
#             original_images_crop = crop_img(original_images)
#
#             # print(original_images.max(), original_images.min(), original_images.mean())
#
#             thresholds = [0.7, 0.8, 1.0, 0.8, 0.7]
#             classified_label, original_exit = Alex_ee_inference.threshold_inference_new(B_alex, 0, original_images,
#                                                                                     thresholds)
#
#             generated_images = generator(original_images)
#             # print(generated_images.max(), generated_images.min(), generated_images.mean())
#             generated_images_crop = crop_img(generated_images)
#
#             #1 use diff to combine
#
#             # diff = generated_images_crop - original_images
#             # diff[:,:, only_edge: -only_edge, only_edge: -only_edge] = 0
#
#             # diff[:, :, :only_edge, :] = 1
#             # diff[:, :, -only_edge:, :] = 1
#             # diff[:, :, :, :only_edge] = 1
#             # diff[:, :, :, -only_edge:] = 1
#             # generated_images_crop += 0.5*diff
#             # generated_images_crop =  diff
#
#             #2 directly combine ori and gen
#
#             # original_images_copy = original_images.clone()
#             # original_images_copy[:,:, only_edge: -only_edge, only_edge: -only_edge] = 2*generated_images_crop[:,:, only_edge: -only_edge, only_edge: -only_edge]
#             # generated_images_crop = original_images_copy
#
#             classified_label_gen, generated_exit = Alex_ee_inference.threshold_inference_new(B_alex, 0, generated_images_crop,
#                                                                                       thresholds)
#
#             print(f"Original Image - label: {classified_label}, \n Exit Location: {original_exit}")
#             print(f"Generated Image - label: {classified_label_gen}, \n Exit Location: {generated_exit}")
#
#             num_forced_class_original += (classified_label == 0).sum().item()
#             num_forced_class_generated += (classified_label_gen == 0).sum().item()
#
#
#             # Show the original and generated images
#             for i in range(num): # show x imgs
#                 plt.figure(figsize=(8, 4))
#                 plt.subplot(1, 2, 1)
#                 plt.title(f"Original Image {i + 1}, "
#                           f"label: {classified_label[i]}, "
#                           f"exit: {original_exit[i]}")
#
#                 out1 = np.transpose(original_images[i].cpu().numpy(), (1, 2, 0))
#                 out1_normalized = (out1 - np.min(out1)) / (np.max(out1) - np.min(out1))
#
#                 plt.imshow(out1_normalized)
#                 plt.subplot(1, 2, 2)
#                 plt.title(f"crop size {crop_size}, replace size {replace_size}, only edge {only_edge}\n "
#                           f"Gen Image {i + 1}, "
#                           f"label: {classified_label_gen[i]}, "
#                           f"exit: {generated_exit[i]}"
#                 )
#
#                 out2 = np.transpose(generated_images_crop[i].cpu().numpy(), (1, 2, 0))
#                 out2_normalized = (out2 - np.min(out2)) / (np.max(out2) - np.min(out2))
#                 plt.imshow(out2_normalized)
#                 plt.show()
#
#         print(f"Number of 0s in Original Labels: {num_forced_class_original}")
#         print(f"Number of 0s in Generated Labels: {num_forced_class_generated}")
#
# if __name__ == "__main__":
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     pl.seed_everything(2024)
#
#     generator = TransformerBlock(dim = 128, heads = 8, mlp_dim = 512, dropout = 0.5).to(device)
#     generator = nn.DataParallel(generator, device_ids=[0, 1, 2, 3])
#
#     B_alex = BranchedAlexNet()
#     B_alex.load_state_dict(torch.load(r"/home/yibo/PycharmProjects/Thesis/weights/B-Alex final/B-Alex_cifar10.pth", weights_only=True))
#     B_alex.to(device)
#
#     root = '/home/yibo/PycharmProjects/Thesis/CIFAR10'
#     dataprep = Data_prep_224_gen(root)
#     train_idx, val_idx, test_idx = dataprep.get_category_index(category = 0) # 0 airplane, 3 cat, 8 ship
#     # print(f"Total entries in train_idx: {len(train_idx)}, val_idx: {len(val_idx)}, test_idx: {len(test_idx)}")
#     trainloader, valloader, testloader = dataprep.create_catogery_loaders(batch_size=256, num_workers=8, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
#     optimizer = optim.Adam(generator.parameters(), lr=1e-4)
#
#     # generator.load_state_dict(torch.load('weights/Generator224/Generator_200_airplane_tanh.pth', weights_only=True))
#     # generator.load_state_dict(torch.load('training_weights/Generator224/Generator_epoch_200.pth', weights_only=True))
#
#     crop_size = 0
#     replace_size = 0
#     only_edge = 0
#
#     train(50)
#     # test(5, 2)  # test n imgs per batch of testloader; in total 4n~5n imgs
#     # gen_dataset(generator, trainloader, valloader, testloader)
#     # # display_gen_dataset('data_split/generated_CIFAR224_test.pkl', 'data_split/generated_CIFAR224_val.pkl', 10)
#     # test_gen_dataset('data_split/generated_CIFAR224_test.pkl', 'data_split/generated_CIFAR224_val.pkl')
#
#
