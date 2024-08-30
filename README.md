# Results from retrained DM's

`generate_images_from_prompts.py` uses a specified diffusion model to generate images. The prompts used to generate images should be provided in a .js file format, such as in the file `evaluation_abstract_data.js`. The repository https://github.com/photosynthesismembrane/caption_generator can be used to generate such a file.

Example execution:
```
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /path/to/diffusion/model --output_folder generated_images
```

`clip_score.py` calculates clip scores from image text combinations and puts these into a `clip_scores.js` file. Generated images from respective DM's are stored in folders `{dataset}_2500_generations_2`, which are not committed to the online github repository for storage reasons. These three folders each contain 18.360 images (8.17gb). Related texts are stored in `evaluation_{dataset}_data.js`. 

The code computes an embedding for each image and caption using the openai/clip-vit-base-patch32 encoder. Cosine similarity is computed between the embeddings. Texts are truncated at 77 tokens, since the retrained DM's do the same for input prompts.

`generate_all.sh` is a script that generates images from evaluation prompts, for each diffusion model ([sd, 8k, 15k, 25k, 35k]x[abstract, landscape, renaissance]). At the end clipscores are calculated on all the generated images. 

`create_clip_score_plots.py` creates plots from distributions of clip scores on many different categories. 