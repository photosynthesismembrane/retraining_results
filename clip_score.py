from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import read_write_json

# Check if CUDA is available and get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the CLIP model and processor and move the model to the GPU
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

similarity_data = []
dataset_names = ["abstract", "landscape", "renaissance"]
max_length = 77  # Max length for CLIP model

# Calculate clip scores for each dataset
for dataset_name in dataset_names:
    discription_data = read_write_json.read_json(f"evaluation_{dataset_name}_data.js", variable_name=f"evaluation_{dataset_name}_data")
    
    # Put all the retrained diffusion model names in a list
    diffusion_models = ["stable_diffusion_v1_5"]
    for model_iterations in [8, 15, 25, 35]:
        for model_name in ["llava", "cogvlm", "deepseek", "best"]:
            diffusion_models.append(f"{dataset_name}_{model_name}_{model_iterations}k")

    vision_models = ["llava", "cogvlm", "deepseek"]
    tasks = ["composition", "balance_elements", "movement", "focus_point", "contrast_elements", "proportion", "foreground_background_4", "symmetry_asymmetry_1", "eye_movement_2"]
    
    for diffusion_model_name in diffusion_models:
        for vision_model_name in vision_models:
            for task_name in tasks:
                for image_name, description_data in discription_data.items():
                    image_path = f"{dataset_name}_2500_generations_2/{diffusion_model_name}_{vision_model_name}_{task_name}_{image_name.replace('.jpg','.png')}"
                    description = description_data[f"{vision_model_name}_answers"][task_name]

                    # Load the image
                    try:
                        image = Image.open(image_path)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        continue

                    try:
                        inputs = processor(
                            text=description,
                            images=[image],  # Repeat image to match text chunks
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=max_length
                        ).to(device)

                        outputs = model(**inputs)
                        image_embeds = outputs.image_embeds
                        text_embeds = outputs.text_embeds
                        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                        similarity = torch.matmul(text_embeds, image_embeds.T)

                        # Calculate average similarity score for the image
                        similarity_scores = similarity.diag().tolist()  # Use the diagonal to get 1:1 scores
                        avg_similarity_score = sum(similarity_scores) / len(similarity_scores)

                        similarity_data.append({
                            "dataset_name": dataset_name,
                            "image_name": image_name,
                            "diffusion_model_name": diffusion_model_name,
                            "vision_model_name": vision_model_name,
                            "task_name": task_name,
                            "similarity_score": avg_similarity_score
                        })

                        # Debug statement to check if data is being added
                        print(f"Processed image {image_name} for task {task_name} with {vision_model_name} on {diffusion_model_name}. Total data length: {len(similarity_data)}")

                    except Exception as e:
                        print(f"Error processing image {image_name} for task {task_name} with {vision_model_name} on {diffusion_model_name}: {e}")

            # Write similarity scores to a JSON file
            try:
                read_write_json.write_json(f"clip_scores.js", similarity_data, variable_name=f"clip_scores_data")
                print("Similarity scores have been written to clip_scores.js")
            except Exception as e:
                print(f"Error writing to JSON file: {e}")
