import argparse
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

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
    
    # Add progress bar for the training loop
    pbar = tqdm(range(num_iterations), desc="Extracting concept vectors")
    for _ in pbar:
        optimizer.zero_grad()
        similarities = model(embeddings)
        loss = -torch.min(similarities)
        loss.backward()
        optimizer.step()
        
        # Update progress bar with current loss
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    return F.normalize(model.concept_vectors.data, dim=1)

def dummy_embedding_creation(image_path, embedding_dim=512):
    """
    A dummy method to create embeddings. Replace this with your actual embedding creation method.
    """
    # This is just a placeholder. Replace with actual embedding creation logic.
    return torch.randn(embedding_dim)

def process_images(directory, embedding_dim=512):
    embeddings = []
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    # Add progress bar for image processing
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(directory, filename)
        embedding = dummy_embedding_creation(image_path, embedding_dim)
        embeddings.append(embedding)
    return torch.stack(embeddings)

def main(args):
    # Process images and create embeddings
    embeddings = process_images(args.directory, args.embedding_dim)
    
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
    parser = argparse.ArgumentParser(description="Extract common concept vectors from images.")
    parser.add_argument("directory", type=str, help="Directory containing the images")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Dimension of the embedding vectors")
    parser.add_argument("--num_concepts", type=int, default=3, help="Number of concept vectors to extract")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations for optimization")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for optimization")
    args = parser.parse_args()
    
    main(args)