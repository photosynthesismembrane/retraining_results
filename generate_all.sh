#!/bin/bash
#SBATCH --time=2-23:59:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100
#SBATCH --job-name=generate
#SBATCH --mem=100G
module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0
module load GCCcore/12.3.0

source ../../venv/bin/activate

export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"
export HF_HOME="/scratch/$USER/.cache/huggingface/transformers"

accelerate config default

huggingface-cli login --token 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

ulimit -s unlimited


python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-best-8k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-llava-8k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-cogvlm-8k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-deepseek-8k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-best-15k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-llava-15k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-cogvlm-15k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-deepseek-15k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-best-25k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-llava-25k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-cogvlm-25k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-deepseek-25k --output_folder abstract_2500_generations

python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-best-8k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-llava-8k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-cogvlm-8k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-deepseek-8k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-best-15k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-llava-15k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-cogvlm-15k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-deepseek-15k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-best-25k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-llava-25k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-cogvlm-25k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-deepseek-25k --output_folder renaissance_2500_generations

python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-best-8k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-llava-8k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-cogvlm-8k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-deepseek-8k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-best-15k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-llava-15k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-cogvlm-15k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-deepseek-15k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-best-25k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-llava-25k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-cogvlm-25k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-deepseek-25k --output_folder landscape_2500_generations

python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path runwayml/stable-diffusion-v1-5 --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path runwayml/stable-diffusion-v1-5 --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path runwayml/stable-diffusion-v1-5 --output_folder abstract_2500_generations

python clip_score.py

deactivate
