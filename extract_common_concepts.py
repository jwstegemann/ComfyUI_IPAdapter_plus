from comfy_script.runtime.real import *
load(args = ComfyUIArgs("--output-directory", "/images/output", "--input-directory", "/images/input", "--disable-metadata", "--reserve-vram", "0.7", "--force-upcast-attention")) #, "--gpu-only", "--force-fp16", "--disable-smart-memory"))
from comfy_script.runtime.real.nodes import *
import time
import torch
import torch.cuda
import comfy.model_management

import argparse
import os
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np


def clip_preprocess(image, size=224):
    mean = torch.tensor([ 0.48145466,0.4578275,0.40821073], device=image.device, dtype=image.dtype)
    std = torch.tensor([0.26862954,0.26130258,0.27577711], device=image.device, dtype=image.dtype)
    image = image.movedim(-1, 1)
    if not (image.shape[2] == size and image.shape[3] == size):
        scale = (size / min(image.shape[2], image.shape[3]))
        image = torch.nn.functional.interpolate(image, size=(round(scale * image.shape[2]), round(scale * image.shape[3])), mode="bicubic", antialias=True)
        h = (image.shape[2] - size)//2
        w = (image.shape[3] - size)//2
        image = image[:,:,h:h+size,w:w+size]
    image = torch.clip((255. * image), 0, 255).round() / 255.0
    return (image - mean.view([3,1,1])) / std.view([3,1,1])

class DummyClipVision:
    clip_vision = None

    def __init__(self):
        # Initialize your CLIP vision model here
        with torch.inference_mode():
            self.clip_vision = CLIPVisionLoader('CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors')
        pass

    def get_embedding(self, file_path):
        torch.cuda.empty_cache()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with torch.inference_mode():
            # Open the image file
            image = Image.open(file_path)
            # Convert the image to RGB mode
            image = image.convert("RGB")
            # Convert the image to a numpy array and normalize
            image_np = np.array(image, dtype=np.float32) / 255.0
            # Convert to a PyTorch tensor and add a batch dimension
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)

            image_tensor = image_tensor.to(self.clip_vision.load_device)
            pixel_values = clip_preprocess(image_tensor).float()

            out = self.clip_vision.model(pixel_values=pixel_values, intermediate_output=-2)
            result = out[1].to(comfy.model_management.intermediate_device())
 #           print("created embedding for ", file_path, " of ", result.shape)
            del image_tensor, pixel_values, out
#            print(torch.cuda.memory_allocated())
            return result

def process_images(directory, clip_model):
    embeddings = []
    
    for filename in tqdm(os.listdir(directory), desc="Processing images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(directory, filename)
            embedding = clip_model.get_embedding(image_path)
            embeddings.append(embedding)
    
    return torch.cat(embeddings, dim=0)

def extract_common_concepts(embeddings, num_concepts=1, num_iterations=1000, learning_rate=0.01, temp=0.07):
    device = embeddings.device
    n, token_count, embedding_dim = embeddings.shape
    
    concept_vectors = torch.randn(num_concepts, embedding_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([concept_vectors], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    
    best_loss = float('inf')
    best_concepts = None
    
    pbar = tqdm(range(num_iterations), desc="Extracting common concepts")
    for i in pbar:
        optimizer.zero_grad()
        
        # Normalize embeddings and concept vectors
        embeddings_norm = F.normalize(embeddings.view(-1, embedding_dim), dim=1).view(n, token_count, embedding_dim)
        concept_vectors_norm = F.normalize(concept_vectors, dim=1)
        
        # Calculate cosine similarity with temperature scaling
        similarities = torch.matmul(embeddings_norm, concept_vectors_norm.T) / temp
        
        # Softmax over concepts for each token
        attention = F.softmax(similarities, dim=2)
        
        # Weighted sum of token embeddings for each concept
        concept_embeddings = torch.matmul(attention.transpose(1, 2), embeddings_norm)
        
        # Maximize similarity between extracted and target concepts
        loss = -F.cosine_similarity(concept_embeddings, concept_vectors_norm, dim=2).mean()
        
        if i % 10 == 0:  # Add some noise every 10 iterations
            loss += 0.01 * torch.randn(1, device=device)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(concept_vectors, max_norm=1.0)
        
        optimizer.step()
        scheduler.step(loss)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_concepts = concept_vectors.clone().detach()
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Best Loss": f"{best_loss:.4f}"})
        
        # Early stopping
#        if i > 100 and loss.item() > best_loss * 1.1:
#            print("Early stopping triggered.")
#            break
    
    return best_concepts


def main(args):
    # Initialize CLIP vision model
    clip_model = DummyClipVision()

    # Process images and create embeddings
    embeddings = process_images(args.image_directory, clip_model)
    
    print(f"Created embeddings of shape: {embeddings.shape}")
    
    # Extract concept vectors
    concept_vectors = extract_common_concepts(
        embeddings, 
        num_concepts=args.num_concepts,
        num_iterations=args.iterations,
        learning_rate=args.learning_rate,
        temp=args.temperature
    )
    
    print(f"Extracted {args.num_concepts} concept vectors of shape: {concept_vectors.shape}")
    
    # Save concept vectors
    torch.save(concept_vectors, 'concept_vectors.pt')
    print("Concept vectors saved to 'concept_vectors.pt'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract common concept vectors from images using CLIP vision.")
    parser.add_argument("image_directory", type=str, help="Directory containing images")
    parser.add_argument("--num_concepts", type=int, default=3, help="Number of concept vectors to extract")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations for optimization")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Initial learning rate for optimization")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for softmax")
    args = parser.parse_args()
    
    main(args)