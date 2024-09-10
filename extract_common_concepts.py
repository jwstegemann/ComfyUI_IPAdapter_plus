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

clip_vision = CLIPVisionLoader('CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors')


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


class ConceptVectorExtractor(nn.Module):
    def __init__(self, embedding_dim, num_concepts=1):
        super().__init__()
        self.concept_vectors = nn.Parameter(torch.randn(num_concepts, embedding_dim))
        
    def forward(self, embeddings):
        embeddings = F.normalize(embeddings, dim=1)
        concept_vectors = F.normalize(self.concept_vectors, dim=1)
        similarities = torch.mm(embeddings, concept_vectors.t())
        return similarities

def extract_concept_vectors(embeddings, num_concepts=1, num_iterations=1000, learning_rate=0.01):
    device = embeddings.device
    model = ConceptVectorExtractor(embeddings.shape[1], num_concepts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    pbar = tqdm(range(num_iterations), desc="Extracting concept vectors")
    for _ in pbar:
        optimizer.zero_grad()
        similarities = model(embeddings)
        loss = -torch.min(similarities)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    return F.normalize(model.concept_vectors.data, dim=1)

def create_embeddings(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Open the image file
    image = Image.open(file_path)
    # Convert the image to RGB mode
    image = image.convert("RGB")
    # Convert the image to a numpy array and normalize
    image_np = np.array(image, dtype=np.float32) / 255.0
    # Convert to a PyTorch tensor and add a batch dimension
    image_tensor = torch.from_numpy(image_np).unsqueeze(0)

    image_tensor = image_tensor.to(clip_vision.load_device)
    pixel_values = clip_preprocess(image_tensor).float()

    out = clip_vision.model(pixel_values=pixel_values, intermediate_output=-2)
    return out[1].to(comfy.model_management.intermediate_device())


def main(args):
    # Load pre-computed embeddings
    embeddings = create_embeddings(args.embeddings_file)
    
    print(f"Loaded embeddings shape: {embeddings.shape}")
    if embeddings.shape != torch.Size([257, 1280]):
        print("Warning: Expected embeddings shape is [257, 1280], but got {embeddings.shape}")
    
    # Extract concept vectors
    concept_vectors = extract_concept_vectors(
        embeddings, 
        num_concepts=args.num_concepts,
        num_iterations=args.iterations,
        learning_rate=args.learning_rate
    )
    
    # Print results
    print(f"Extracted {args.num_concepts} concept vectors of shape: {concept_vectors.shape}")
    
    # Save concept vectors
    torch.save(concept_vectors, 'concept_vectors.pt')
    print("Concept vectors saved to 'concept_vectors.pt'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract common concept vectors from pre-computed embeddings.")
    parser.add_argument("embeddings_file", type=str, help="File containing pre-computed embeddings")
    parser.add_argument("--num_concepts", type=int, default=3, help="Number of concept vectors to extract")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations for optimization")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for optimization")
    args = parser.parse_args()
    
    main(args)