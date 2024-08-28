import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, f_oneway
import matplotlib.font_manager as font_manager

# Example data: Three groups with sample data
data_dict = {
    'group1': [23, 20, 22, 21, 24, 20, 22],
    'group2': [30, 32, 29, 28, 33, 30, 31],
    'group3': [18, 19, 17, 20, 18, 17, 19]
}

def plot_gaussians(data_dict, plot_title, x_label, y_label, file_name, display_numbers=True):
    try:
        # Filter out groups with insufficient data or NaN variances
        filtered_data_dict = {k: v for k, v in data_dict.items() if len(v) > 1 and pd.Series(v).var() > 0}

        if not filtered_data_dict or len(filtered_data_dict) < 2:
            raise ValueError(f"Not enough data to plot distributions for {plot_title}.")

        # Create a DataFrame from the data dictionary
        data = {'value': [], 'group': []}
        for group, values in filtered_data_dict.items():
            data['value'].extend(values)
            data['group'].extend([group] * len(values))

        df = pd.DataFrame(data)

        # Calculate means and variances
        means = df.groupby('group')['value'].mean()
        variances = df.groupby('group')['value'].var()

        # Identify the groups with the minimum and maximum means
        min_mean_group = means.idxmin()
        max_mean_group = means.idxmax()

        # Calculate x_min and x_max based on the specific variances
        x_min = means[min_mean_group] - 3 * np.sqrt(variances[min_mean_group])
        x_max = means[max_mean_group] + 3 * np.sqrt(variances[max_mean_group])

        # Generate x-values
        x = np.linspace(x_min, x_max, 1000)

        # Perform one-way ANOVA
        groups = [values for values in filtered_data_dict.values()]
        F_statistic, p_value = f_oneway(*groups)

        # Plotting the Gaussian distributions
        plt.figure(figsize=(10, 6))

        for group in means.index:
            mean = means[group]
            variance = variances[group]
            std_dev = np.sqrt(variance)
            y = norm.pdf(x, mean, std_dev)
            mean_variance_text = f"µ: {mean:.3f} σ: {std_dev:.3f}" if display_numbers else ''
            space = ' ' * (24 - len(group))
            plt.plot(x, y, label=f'{group}{space}{mean_variance_text}')

        # Add ANOVA results outside the plot
        plt.figtext(0.91, 0.1, f'One-way ANOVA:\np-value: {p_value:.4f}',
                    horizontalalignment='right', fontsize=20,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))


        font = font_manager.FontProperties(family='Courier New',
                                        weight='normal',
                                        style='normal', size=20)

        # Add titles and labels
        plt.title(f"{plot_title}", fontsize=20)
        plt.xlabel(f"{x_label}", fontsize=20)
        plt.ylabel(f"{y_label}", fontsize=20)
        plt.legend(prop=font)

        # Show the plot
        plt.grid(True)

        # Save the plot
        plt.savefig(f"{file_name}.png")

    except Exception as e:
        # Generate an empty plot with error message
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'{str(e)}', fontsize=20, ha='center')
        # plt.title('Error in Plot Generation')
        plt.grid(False)
        plt.axis('off')

        # Save the error plot
        plt.savefig(f"{file_name}.png")

    finally:
        # Close the plot
        plt.close()

# Call the function
plot_gaussians(data_dict, 'Gaussian Distributions', 'Value', 'Density', 'gaussian_plot')
