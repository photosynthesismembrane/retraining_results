import read_write_json
import plot_gaussians

# Read the data from the JSON file
data = read_write_json.read_json("clip_scores.js",variable_name="clip_scores_data")

# Make output folder if it doesn't exist
import os
os.makedirs('clip_score_plots', exist_ok=True)

iterations = ["8k", "15k", "25k", "35k"]

for iteration in iterations:
    vision_models = ["llava", "cogvlm", "deepseek"]
    tasks = ["composition", "balance_elements", "movement", "focus_point", "contrast_elements", "proportion", "foreground_background_4", "symmetry_asymmetry_1", "eye_movement_2"]
    dataset_names = ["abstract", "landscape", "renaissance"]
    for dataset_name in dataset_names:
        diffusion_models = ["stable_diffusion_v1_5", f"{dataset_name}_llava_{iteration}", f"{dataset_name}_cogvlm_{iteration}", f"{dataset_name}_deepseek_{iteration}", f"{dataset_name}_best_{iteration}"]
        for vision_model_name in vision_models:
            for task_name in tasks:
                data_dict = {}
                for diffusion_model_name in diffusion_models:
                    data_dict[diffusion_model_name] = [item["similarity_score"] for item in data if item["dataset_name"] == dataset_name and item["vision_model_name"] == vision_model_name and item["task_name"] == task_name and item["diffusion_model_name"] == diffusion_model_name]
                
                plot_gaussians.plot_gaussians(data_dict, f"{dataset_name.capitalize()} {vision_model_name} {task_name.split('_')[0]}", 'Clip score', 'Density', f'clip_score_plots\\{iteration}_{dataset_name}_{vision_model_name}_{task_name}_clip_score')


    for dataset_name in dataset_names:
        diffusion_models = ["stable_diffusion_v1_5", f"{dataset_name}_llava_{iteration}", f"{dataset_name}_cogvlm_{iteration}", f"{dataset_name}_deepseek_{iteration}", f"{dataset_name}_best_{iteration}"]
        for vision_model_name in vision_models:
            data_dict = {}
            for diffusion_model_name in diffusion_models:
                data_dict[diffusion_model_name] = [item["similarity_score"] for item in data if item["dataset_name"] == dataset_name and item["vision_model_name"] == vision_model_name and item["diffusion_model_name"] == diffusion_model_name]
                
            plot_gaussians.plot_gaussians(data_dict, f'{dataset_name.capitalize()} {vision_model_name}', 'Clip score', 'Density', f'clip_score_plots\\{iteration}_{dataset_name}_{vision_model_name}_clip_score')

    for dataset_name in dataset_names:
        diffusion_models = ["stable_diffusion_v1_5", f"{dataset_name}_llava_{iteration}", f"{dataset_name}_cogvlm_{iteration}", f"{dataset_name}_deepseek_{iteration}", f"{dataset_name}_best_{iteration}"]
        data_dict = {}
        for diffusion_model_name in diffusion_models:
            data_dict[diffusion_model_name] = [item["similarity_score"] for item in data if item["dataset_name"] == dataset_name and item["diffusion_model_name"] == diffusion_model_name]
            
        plot_gaussians.plot_gaussians(data_dict, f'{dataset_name.capitalize()}', 'Clip score', 'Density', f'clip_score_plots\\{iteration}_{dataset_name}_clip_score')

    for dataset_name in dataset_names:
        diffusion_models = ["stable_diffusion_v1_5", f"{dataset_name}_llava_{iteration}", f"{dataset_name}_cogvlm_{iteration}", f"{dataset_name}_deepseek_{iteration}", f"{dataset_name}_best_{iteration}"]
        for task_name in tasks:
            data_dict = {}
            for diffusion_model_name in diffusion_models:
                data_dict[diffusion_model_name] = [item["similarity_score"] for item in data if item["dataset_name"] == dataset_name and item["task_name"] == task_name and item["diffusion_model_name"] == diffusion_model_name]
                
            plot_gaussians.plot_gaussians(data_dict, f'{dataset_name.capitalize()} {task_name.split("_")[0]}', 'Clip score', 'Density', f'clip_score_plots\\{iteration}_{dataset_name}_{task_name}_clip_score')

    data_dict = {}
    for dataset_name in dataset_names:
        diffusion_models = ["stable_diffusion_v1_5", f"{dataset_name}_llava_{iteration}", f"{dataset_name}_cogvlm_{iteration}", f"{dataset_name}_deepseek_{iteration}", f"{dataset_name}_best_{iteration}"]
        for diffusion_model_name in diffusion_models:
            data_dict[diffusion_model_name.replace(f'{dataset_name}_','')] = [item["similarity_score"] for item in data if item["diffusion_model_name"] == diffusion_model_name]

    plot_gaussians.plot_gaussians(data_dict, 'All', 'Clip score', 'Density', f'clip_score_plots\\{iteration}_diffusion_models_clip_score')


    for task_name in tasks:
        data_dict = {}
        for dataset_name in dataset_names:
            diffusion_models = ["stable_diffusion_v1_5", f"{dataset_name}_llava_{iteration}", f"{dataset_name}_cogvlm_{iteration}", f"{dataset_name}_deepseek_{iteration}", f"{dataset_name}_best_{iteration}"]
            for diffusion_model_name in diffusion_models:
                data_dict[diffusion_model_name.replace(f'{dataset_name}_','')] = [item["similarity_score"] for item in data if item["diffusion_model_name"] == diffusion_model_name]

        plot_gaussians.plot_gaussians(data_dict, f'{task_name.split("_")[0]}', 'Clip score', 'Density', f'clip_score_plots\\{iteration}_{task_name}_clip_score')

    for vision_model_name in vision_models:
        data_dict = {}
        for dataset_name in dataset_names:
            diffusion_models = ["stable_diffusion_v1_5", f"{dataset_name}_llava_{iteration}", f"{dataset_name}_cogvlm_{iteration}", f"{dataset_name}_deepseek_{iteration}", f"{dataset_name}_best_{iteration}"]
            for diffusion_model_name in diffusion_models:
                data_dict[diffusion_model_name.replace(f'{dataset_name}_','')] = [item["similarity_score"] for item in data if item["vision_model_name"] == vision_model_name and item["diffusion_model_name"] == diffusion_model_name]

        plot_gaussians.plot_gaussians(data_dict, f'{vision_model_name.capitalize()}', 'Clip score', 'Density', f'clip_score_plots\\{iteration}_{vision_model_name}_clip_score')

    for vision_model_name in vision_models:
        for task_name in tasks:
            data_dict = {}
            for dataset_name in dataset_names:
                diffusion_models = ["stable_diffusion_v1_5", f"{dataset_name}_llava_{iteration}", f"{dataset_name}_cogvlm_{iteration}", f"{dataset_name}_deepseek_{iteration}", f"{dataset_name}_best_{iteration}"]
                for diffusion_model_name in diffusion_models:
                    data_dict[diffusion_model_name.replace(f'{dataset_name}_','')] = [item["similarity_score"] for item in data if item["vision_model_name"] == vision_model_name and item["task_name"] == task_name and item["diffusion_model_name"] == diffusion_model_name]

            plot_gaussians.plot_gaussians(data_dict, f'{vision_model_name.capitalize()} {task_name.split("_")[0]}', 'Clip score', 'Density', f'clip_score_plots\\{iteration}_{vision_model_name}_{task_name}_clip_score')
        data_dict = {}
        

    html = f"""
    <div class="container">
        <div class="category">
            <h2>All</h2>
            <div class="plots">
                <img src="clip_score_plots/{iteration}_diffusion_models_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_llava_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_cogvlm_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_deepseek_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_llava_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_llava_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_llava_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_llava_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_llava_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_llava_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_llava_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_llava_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_llava_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_cogvlm_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_cogvlm_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_cogvlm_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_cogvlm_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_cogvlm_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_cogvlm_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_cogvlm_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_cogvlm_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_cogvlm_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_deepseek_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_deepseek_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_deepseek_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_deepseek_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_deepseek_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_deepseek_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_deepseek_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_deepseek_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_deepseek_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
            </div>
        </div>
        <div class="category">
            <h2>Abstract</h2>
            <div class="plots">
                <img src="clip_score_plots/{iteration}_abstract_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_llava_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_cogvlm_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_deepseek_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_llava_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_llava_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_llava_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_llava_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_llava_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_llava_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_llava_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_llava_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_llava_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_cogvlm_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_cogvlm_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_cogvlm_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_cogvlm_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_cogvlm_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_cogvlm_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_cogvlm_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_cogvlm_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_cogvlm_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_deepseek_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_deepseek_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_deepseek_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_deepseek_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_deepseek_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_deepseek_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_deepseek_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_deepseek_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_abstract_deepseek_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
            </div>
        </div>
        <div class="category">
            <h2>Landscape</h2>
            <div class="plots">
                <img src="clip_score_plots/{iteration}_landscape_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_llava_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_cogvlm_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_deepseek_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_llava_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_llava_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_llava_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_llava_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_llava_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_llava_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_llava_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_llava_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_llava_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_cogvlm_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_cogvlm_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_cogvlm_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_cogvlm_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_cogvlm_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_cogvlm_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_cogvlm_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_cogvlm_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_cogvlm_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_deepseek_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_deepseek_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_deepseek_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_deepseek_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_deepseek_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_deepseek_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_deepseek_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_deepseek_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_landscape_deepseek_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
            </div>
        </div>
        <div class="category">
            <h2>Renaissance</h2>
            <div class="plots">
                <img src="clip_score_plots/{iteration}_renaissance_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_llava_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_cogvlm_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_deepseek_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_llava_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_llava_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_llava_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_llava_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_llava_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_llava_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_llava_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_llava_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_llava_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_cogvlm_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_cogvlm_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_cogvlm_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_cogvlm_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_cogvlm_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_cogvlm_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_cogvlm_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_cogvlm_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_cogvlm_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_deepseek_composition_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_deepseek_balance_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_deepseek_movement_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_deepseek_focus_point_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_deepseek_contrast_elements_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_deepseek_proportion_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_deepseek_foreground_background_4_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_deepseek_symmetry_asymmetry_1_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
                <img src="clip_score_plots/{iteration}_renaissance_deepseek_eye_movement_2_clip_score.png" style="width: 100%;" onclick="openModal(this.src)">
            </div>
        </div>
    </div>
    """

    with open(f'clip_score_plots/{iteration}_clip_score_plots.html', 'w') as f:
        f.write(html)
