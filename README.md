# Results from retrained DM's

`clip_score.py` calculates clip scores from image text combinations and puts these into a `clip_scores.js` file. Generated images from respective DM's are stored in folders `{dataset}_2500_generations_2`, which are not committed to the online github repository for storage reasons. These three folders each contain 18.360 images (8.17gb). Related texts are stored in `evaluation_{dataset}_data.js`. 

The code computes an embedding for each image and caption using the openai/clip-vit-base-patch32 encoder. Cosine similarity is computed between the embeddings. Texts are truncated at 77 tokens, since the retrained DM's do the same for input prompts.

`create_clip_score_plots.py` creates plots from distributions of clip scores on many different categories. 