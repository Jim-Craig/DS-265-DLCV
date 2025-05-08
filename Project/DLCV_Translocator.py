import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score
import warnings
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
import gc
from sklearn.preprocessing import LabelEncoder
import joblib
from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings("ignore", message="numerical errors at iteration 0")

#Constant
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Semantic Segmentation Model
class SemanticSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.segmentation_model = deeplabv3_resnet50(pretrained=True)
        # self.segmentation_model.eval()  # Freeze weights

    def forward(self, x):
        with torch.no_grad():
            seg_map = self.segmentation_model(x)['out']
        return seg_map

#Multimodal Feature Fusion (MFF)
class MultimodalFusion(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.projection = nn.Linear(embed_dim, embed_dim)  # f(.) projection
        self.back_projection = nn.Linear(embed_dim, embed_dim)  # g(.) back-projection

        # Attention module
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=1)  # Generates [w_rgb, w_seg]
        )

    def forward(self, rgb_cls, seg_cls):
        # Compute modality attention
        att_input = torch.cat([rgb_cls, seg_cls], dim=1)
        weights = self.attention_mlp(att_input)  # [batch, 2]
        w_rgb, w_seg = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)

        # Weighted CLS token fusion (final layer)
        rgb_final = (1 + w_rgb) * rgb_cls
        seg_final = (1 + w_seg) * seg_cls
        fmm = torch.cat([rgb_final, seg_final], dim=1)  # Final fused feature

        return fmm
    
class PositionalEncoder(nn.Module):
    def __init__(self, dim_model: int, dropout_p: float = 0.1, max_len: int=1000):
        """Initializes the positional embedding layer to enrich data fed into transformers
           with positional information.
        Args:
            dim_model (int): model dimension
            dropout_p (float, optional): dropout for all embeddings. Defaults to 0.1.
            max_len (int, optional): determines how far the position can influence other tokens. Defaults to 1000.
        Note:
            This code is a modified version of: `<https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`_.
        """
        super().__init__()

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Encoding
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pos_encoding', nn.Parameter(pos_encoding, requires_grad=False))

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        """Generates positional embeddings.
        Args:
            token_embedding (torch.tensor): original embeddings
        Returns:
            torch.tensor: transformed embeddings
        """
        # Residual connection + positional encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
    
class MultimodalTransformer(nn.Module):
    def __init__(self, img_size, patch_size = 8, vit_model: str = "vit_base_patch16_224", embed_dim: int = 768, num_layers: int = 3):
        super(MultimodalTransformer, self).__init__()

        # Load pre-trained ViT models for RGB and Semantic maps
        self.rgb_transformer = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=num_layers, num_heads=6, mlp_ratio=4)
        self.seg_transformer = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, in_chans=1, depth=num_layers, num_heads=6, mlp_ratio=4)

        # Projection layers to align CLS token dimensions
        self.cls_projection = nn.Linear(embed_dim, embed_dim)  # f(.) projection
        self.back_projection = nn.Linear(embed_dim, embed_dim)  # g(.) back-projection

        # Attention fusion of CLS token from the last layer
        self.fusion_module = MultimodalFusion()

        #Positional Encoder
        self.pos_encoder = PositionalEncoder(dim_model=embed_dim)


    def forward(self, rgb_input, seg_input):
        # Extract embeddings from both transformers
        rgb_tokens = self.rgb_transformer.patch_embed(rgb_input)
        seg_tokens = self.seg_transformer.patch_embed(seg_input)

        cls_rgb = self.rgb_transformer.cls_token.expand(rgb_tokens.shape[0], -1, -1)
        cls_seg = self.seg_transformer.cls_token.expand(seg_tokens.shape[0], -1, -1)

        # Positional embedding
        #Concat the cls token for resective tokens(rgb or seg)
        rgb_tokens = torch.cat([cls_rgb, rgb_tokens], dim=1)
        seg_tokens = torch.cat([cls_seg, seg_tokens], dim=1)
        #add the positional embeddings
        rgb_tokens = self.pos_encoder(rgb_tokens)
        seg_tokens = self.pos_encoder(seg_tokens)

        for layer in range(len(self.rgb_transformer.blocks)):
            rgb_tokens = self.rgb_transformer.blocks[layer](rgb_tokens)
            seg_tokens = self.seg_transformer.blocks[layer](seg_tokens)

            # Extract CLS tokens after each layer
            cls_rgb = self.cls_projection(rgb_tokens[:, 0])
            cls_seg = self.cls_projection(seg_tokens[:, 0])

            # Sum CLS tokens and append back to patch tokens
            fused_cls = self.back_projection(cls_rgb + cls_seg)
            rgb_tokens = torch.cat([fused_cls.unsqueeze(1), rgb_tokens[:, 1:]], dim=1)
            seg_tokens = torch.cat([fused_cls.unsqueeze(1), seg_tokens[:, 1:]], dim=1)

        # Final CLS tokens from last layer
        cls_rgb = rgb_tokens[:, 0]
        cls_seg = seg_tokens[:, 0]

        # Attention fusion of the cls tokens from the last layer
        fmm = self.fusion_module(cls_rgb, cls_seg)

        return fmm

#Define a mlp layer that predict a class based on the input
class mlp(nn.Module):
  def __init__(self, input_dim = 1536, output_dim = 10):
    super(mlp, self).__init__()
    self.fc1 = nn.Linear(input_dim, 1024)
    self.fc2 = nn.Linear(1024, 2048)
    self.fc3 = nn.Linear(2048, output_dim)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc3(x)
    return x
  
#Define a classifier for based on the Semantic Segmentor and multimodal model.
class MultiModalClassifier(nn.Module):
  def __init__(self, input_dim = 256,feature_size = 1536,  output_dim = 2781):
    super(MultiModalClassifier, self).__init__()
    self.sematic_segmantic = SemanticSegmentor()
    self.multimodal = MultimodalTransformer(input_dim)
    self.mlp = mlp(feature_size, output_dim)

  # x is the rgb image of the shape (batch_size, channels, height, width)
  def forward(self, x):
    #Generate the semantic maps for the input images

    semantic_output = self.sematic_segmantic(x)
    semantic_map = torch.argmax(semantic_output.squeeze(), dim=1).unsqueeze(1).float()

    # Pass the semantic map and the rgb images through the MultimodalTransformer
    multimodal_output = self.multimodal(x, semantic_map)# (batch, 1536)

    # pass throught the mlp to get the classes
    output = self.mlp(multimodal_output)
    return output
  
def denormalize(img_tensor):
    """Reverse normalization using ImageNet stats"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor * std + mean

class ImageLabelDataset(Dataset):
    def __init__(self, image_dir, dataframe, transform=None):
        self.image_dir = image_dir
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_filename = row['filename']
        label = row['polygon_label']
        img_path = os.path.join(self.image_dir, img_filename)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
    
def getDataset():
    img_path = '/home/godwinkhalko/DLCV/00'
    label_path = '/home/godwinkhalko/DLCV/labelled_points.xlsx'
    
    df = pd.read_excel(label_path, dtype={'id': str})
    
    image_files = os.listdir(img_path)
    id_to_filename = {}
    for f in image_files:
        id_ = os.path.splitext(f)[0] 
        id_to_filename[id_] = f

    label_encoder = LabelEncoder()

    df['polygon_label'] = label_encoder.fit_transform(df['polygon_label'])

    joblib.dump(label_encoder, 'label_encoder.pkl')

    filtered_df = df[df['id'].isin(id_to_filename.keys())]

    filtered_df['filename'] = filtered_df['id'].map(id_to_filename)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageLabelDataset(image_dir=img_path, dataframe=filtered_df, transform=transform)
    return dataset


def showImages(dataloader):
    images, labels = next(iter(dataloader))
    
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    for i in range(5):
        image = denormalize(images[i]).cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image, 0, 1)
        
        axs[i].imshow(image)
        axs[i].set_title(f"Label: {labels[i].item()}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

def train(model, dataset, epochs, batch_size, optimizer, criterion, scheduler, save_point = 500):
    device = next(model.parameters()).device

    # Define the Dataloader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    train_loss = []
    train_accuracy = []
    checkpoint_path = "/home/godwinkhalko/DLCV/checkpoint.pth"
    print(f"Checking if a checkpoint exists")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting from scratch.")
        start_epoch = 0

    print("Started Training")
    for epoch in range(start_epoch, epochs):
        model.train()  
        optimizer.zero_grad()  

        training_loss_batch = []
        training_accuracies_batch = []

        print(f"Started training for {epoch + 1}")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  

            # Forward pass
            output= model(data)

            loss = criterion(output, target) 

            training_loss_batch.append(loss.item())

            # Backpropagation
            loss.backward()

            optimizer.step()  
            scheduler.step()

            # Calculate accuracy
            predicted_classes = torch.argmax(output, dim=1)
            accuracy = accuracy_score(target.cpu().numpy(), predicted_classes.cpu().numpy())
            training_accuracies_batch.append(accuracy)

            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}, Accuracy: {accuracy}", end="\r")
            if batch_idx % save_point == 0:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
            
        # Average training loss and accuracy
        train_loss.append(np.mean(training_loss_batch))
        train_accuracy.append(np.mean(training_accuracies_batch))
        print(f"\n Epoch {epoch+1}/{epochs}, Training Loss: {np.mean(training_loss_batch)}, Training Accuracy: {np.mean(training_accuracies_batch)}")
    
    return model

class TestImageLabelDataset(Dataset):
    def __init__(self, image_dir, dataframe, transform=None):
        self.image_dir = image_dir
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_filename = row['filename']
        img_path = os.path.join(self.image_dir, img_filename)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        
        return image, row['id'], row['latitude'], row['longitude']

if __name__ == "__main__":
    # #Initialize the training parameters
    # epochs = 10
    # batch_size = 16
    # learning_rate = 1e-2
    # weight_decay = 1e-5

    # #Initlaize the models and stuff
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MultiModalClassifier()
    # model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    # criterion = nn.CrossEntropyLoss()
    # train_dataset = getDataset()

    # #Run the training
    # trained_model = train(model=model,
    #                     dataset=train_dataset,
    #                     epochs=epochs,
    #                     batch_size=batch_size,
    #                     optimizer=optimizer,
    #                     criterion=criterion,
    #                     scheduler=scheduler,
    #                     save_point=50)
    # torch.save(trained_model.state_dict(), "/home/godwinkhalko/DLCV/Trained_Model.pth")
    # del model
    # del optimizer
    # del criterion
    # del train_dataset
    # del trained_model

    label_path = '/home/godwinkhalko/DLCV/test.csv'
    img_path = '/home/godwinkhalko/DLCV/00'
    print("Creating the Dataframe")
    df = pd.read_csv(label_path, dtype={'id': str})

    image_files = os.listdir(img_path)
    id_to_filename = {}
    for f in image_files:
        id_ = os.path.splitext(f)[0] 
        id_to_filename[id_] = f

    filtered_df = df[df['id'].isin(id_to_filename.keys())]
    filtered_df['filename'] = filtered_df['id'].map(id_to_filename)
    filtered_df_mod = filtered_df[["id", "latitude", "longitude", "filename"]]
    print("Created the Dataframe")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Model
    print("Loading the model")
    model = MultiModalClassifier()

    checkpoint = torch.load("/home/godwinkhalko/DLCV/Trained_Model.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    print("Finished Loading the model")
    model.eval()
    results = []
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    print("Creating the datasets")
    dataset = TestImageLabelDataset(image_dir="/home/godwinkhalko/DLCV/00", dataframe=filtered_df_mod, transform=transform)
    # subset =Subset(dataset, list(range(10)))

    test_loader = DataLoader(dataset=dataset, batch_size=32)
    print("Creating the dataloader")
    with torch.no_grad():
        for batch_idx, (ipnuts, ids, lats, lons) in enumerate(test_loader):
            # Assume each batch has the following
            # batch = (inputs, (id, lat, lon))
            print(f"Processing Batch: {batch_idx}/{len(test_loader)}", end="\r")
            inputs =ipnuts.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()  # shape: [batch_size, num_classes]

            for i in range(len(ids)):
                results.append({
                    "id": ids[i],
                    "latitude": lats[i],
                    "longitude": lons[i],
                    "softmax_probs": probs[i]  # This will be a NumPy array
                })

    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv("test_results.csv", index=False)

    gc.collect()
    torch.cuda.empty_cache()