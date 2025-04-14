import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from vit_face_detector import VisionTransformer
from utils import (
    visualize_patches,
    visualize_attention,
    inspect_positional_embeddings,
)


def analyze_vision_transformer(input_path: str, output_path: str):
    # Load the model
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout=0.1,
        drop_path=0.1
    )

    # Load the trained weights
    model.load_state_dict(torch.load(
        f'{output_path}/best_model.pth', map_location=torch.device('cpu')))
    model.eval()

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # 1. Analyze Patch Embeddings
    sample_image = Image.open(input_path).convert('RGB')
    transformed_image = transform(sample_image)
    patch_vis = visualize_patches(transformed_image, patch_size=16)
    plt.savefig(f'{output_path}/plots/patch_visualization.png')
    plt.close()

    # 2. Analyze Attention Patterns
    # Look at different layers and heads
    layers_to_analyze = [0, 2, 4]  # Early, middle, and late layers
    heads_to_analyze = [0, 1, 2]   # Different attention heads

    for layer_idx in layers_to_analyze:
        for head_idx in heads_to_analyze:
            attn_vis = visualize_attention(
                model, transformed_image, head_idx=head_idx, layer_idx=layer_idx)
            plt.savefig(
                f'{output_path}/plots/attention_l{layer_idx}_h{head_idx}.png')
            plt.close()

    # 3. Analyze Positional Embeddings
    pos_embed_vis = inspect_positional_embeddings(model)
    plt.savefig(f'{output_path}/plots/positional_embeddings.png')
    plt.close()

    # # 4. Test on sample images
    # test_images = [
    #     '../ClassifiedData/facingAwayImages/image00004.jpg',
    #     '../ClassifiedData/facingAwayImages/image00006.jpg',
    #     '../ClassifiedData/facingAwayImages/image00010.jpg',
    #     '../ClassifiedData/facingAwayImages/image00022.jpg',
    #     '../ClassifiedData/frontFacingImages/image00002.jpg',
    #     '../ClassifiedData/frontFacingImages/image00008.jpg',
    #     '../ClassifiedData/frontFacingImages/image00013.jpg',
    #     '../ClassifiedData/frontFacingImages/image00026.jpg',
    # ]

    results = []
    # for img_path in test_images:
    #     pred, conf = test_model(model, img_path, transform)
    #     results.append({
    #         'image': img_path,
    #         'prediction': 'Face' if pred == 1 else 'Non-face',
    #         'confidence': conf
    #     })

    return results


if __name__ == "__main__":
    input_path = '../ClassifiedData/frontFacingImages/image00014.jpg'
    output_path = "../results/vit_v1"
    results = analyze_vision_transformer(input_path, output_path)
