import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def create_heatmaps(selected_models):
    # Create output directories for visualizations
    heatmap_dir = 'noise_heatmap_selected'
    heatmap_pdf_dir = os.path.join(heatmap_dir, 'pdf')

    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(heatmap_pdf_dir, exist_ok=True)

    # Load the CSV files
    df_all_results = pd.read_csv('../results/model_comparison_results.csv')
    df_baseline_results = pd.read_csv('../results/baseline_comparison_results.csv')

    # Create heatmaps for each dataset showing noise impact on selected model performance
    for dataset_name in ['breast_cancer', 'breast_cancer_pca', 'iris', 'iris_pca', 'titanic', 'titanic_pca', 'wine', 'wine_pca']:
        dataset_results = df_all_results[df_all_results['Dataset'] == dataset_name]

        # Skip if dataset not found
        if dataset_results.empty:
            print(f"No data found for {dataset_name}, skipping...")
            continue

        # Get best performance for each noise combination, model, and kernel
        best_per_combo = dataset_results.loc[
            dataset_results.groupby(['Feature_Noise', 'Label_Noise', 'Model', 'Kernel'])['Test Acc'].idxmax()
        ]
        
        # Load baseline results for this dataset
        baseline_dataset_results = df_baseline_results[df_baseline_results['Dataset'] == dataset_name]
        
        # Get best baseline performance for each noise combination
        baseline_models_data = []
        if not baseline_dataset_results.empty:
            # KNN
            knn_results = baseline_dataset_results[baseline_dataset_results['Model'] == 'KNN'].copy()
            if not knn_results.empty:
                knn_best = knn_results.loc[
                    knn_results.groupby(['Feature_Noise', 'Label_Noise'])['Test Acc'].idxmax()
                ]
                knn_best['Model'] = 'KNN'
                baseline_models_data.append(knn_best)
            
            # Decision Tree - gini
            dt_gini_results = baseline_dataset_results[baseline_dataset_results['Model'] == 'DecisionTree-gini'].copy()
            if not dt_gini_results.empty:
                dt_gini_best = dt_gini_results.loc[
                    dt_gini_results.groupby(['Feature_Noise', 'Label_Noise'])['Test Acc'].idxmax()
                ]
                dt_gini_best['Model'] = 'Decision Tree (gini)'
                baseline_models_data.append(dt_gini_best)
            
            # Decision Tree - entropy
            dt_entropy_results = baseline_dataset_results[baseline_dataset_results['Model'] == 'DecisionTree-entropy'].copy()
            if not dt_entropy_results.empty:
                dt_entropy_best = dt_entropy_results.loc[
                    dt_entropy_results.groupby(['Feature_Noise', 'Label_Noise'])['Test Acc'].idxmax()
                ]
                dt_entropy_best['Model'] = 'Decision Tree (entropy)'
                baseline_models_data.append(dt_entropy_best)
    
        # Create separate figures for each kernel
        for kernel in ['linear', 'rbf']:
            # Determine grid layout based on number of models
            n_models = len(selected_models)
            
            # Create 1xN layout
            fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
            if n_models == 1:
                axes = [axes]
            else:
                axes = axes.ravel()
            
            im = None  # Store the last image for colorbar
            
            # Collect all values to find global min/max for this kernel
            all_values = []
            for model in selected_models:
                # Check if this is a baseline model
                if model in ['KNN', 'Decision Tree (gini)', 'Decision Tree (entropy)']:
                    if not baseline_models_data:
                        continue
                    # Use baseline data
                    baseline_combined = pd.concat(baseline_models_data, ignore_index=True)
                    model_data = baseline_combined[baseline_combined['Model'] == model]
                else:
                    # Use SVM data
                    model_data = best_per_combo[(best_per_combo['Model'] == model) & 
                                            (best_per_combo['Kernel'] == kernel)]
                
                pivot = model_data.pivot_table(
                    values='Test Acc',
                    index='Label_Noise',
                    columns='Feature_Noise',
                    aggfunc='mean'
                )
                all_values.extend(pivot.values.flatten())
            
            # Remove NaN values and calculate min/max
            all_values = [v for v in all_values if not np.isnan(v)]
            vmin = min(all_values) if all_values else 0.5
            vmax = max(all_values) if all_values else 1.0

            for idx, model in enumerate(selected_models):
                # Check if this is a baseline model
                if model in ['KNN', 'Decision Tree (gini)', 'Decision Tree (entropy)']:
                    if not baseline_models_data:
                        axes[idx].set_visible(False)
                        continue
                    # Use baseline data
                    baseline_combined = pd.concat(baseline_models_data, ignore_index=True)
                    model_data = baseline_combined[baseline_combined['Model'] == model]
                else:
                    # Use SVM data
                    model_data = best_per_combo[(best_per_combo['Model'] == model) & 
                                            (best_per_combo['Kernel'] == kernel)]
                
                # Create pivot table
                pivot = model_data.pivot_table(
                    values='Test Acc',
                    index='Label_Noise',
                    columns='Feature_Noise',
                    aggfunc='mean'
                )

                # Reorder columns and index
                feature_order = ['Clean', '5%', '10%', '15%', '20%']
                label_order = ['0%', '5%', '10%', '15%', '20%']
                pivot = pivot.reindex(index=label_order, columns=feature_order)

                # Plot heatmap
                ax = axes[idx]
                im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)

                # Set ticks
                ax.set_xticks(range(len(feature_order)))
                ax.set_yticks(range(len(label_order)))
                ax.set_xticklabels(feature_order)
                ax.set_yticklabels(['Clean' if label == '0%' else label for label in label_order])

                # Add text annotations
                for i in range(len(label_order)):
                    for j in range(len(feature_order)):
                        if not np.isnan(pivot.values[i, j]):
                            value = pivot.values[i, j]
                            text = ax.text(j, i, f'{value*100:.1f}%',
                                        ha='center', va='center', color='black', fontsize=14)

                # if model == "SKiP-average":
                #     display_model = "SKiP"
                if model == "SKiP-multiply-minmax":
                    display_model = "SKiP-minmax-multiply"
                elif model == "SKiP-average-minmax":
                    display_model = "SKiP-minmax-average"
                else:
                    display_model = model

                ax.set_title(f'{display_model}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Feature Outlier', fontsize=10)
                ax.set_ylabel('Label Outlier', fontsize=10)

            # Add colorbar to 1xN layout
            if im is not None:
                cbar_ax = fig.add_axes([0.92, 0.13, 0.01, 0.73])  
                cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='Test Accuracy')
                cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))

            # plt.suptitle(f'Noise Impact on Test Accuracy - {dataset_name.upper()} ({kernel.upper()} Kernel)', 
            #             fontsize=15, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 0.91, 0.95])
            output_path_png = os.path.join(heatmap_dir, f'noise_heatmap_{dataset_name}_{kernel}.png')
            output_path_pdf = os.path.join(heatmap_pdf_dir, f'noise_heatmap_{dataset_name}_{kernel}.pdf')
            plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_path_pdf, bbox_inches='tight')
            plt.close()  # Close figure to free memory
            print(f"Saved heatmap (1xN) for {dataset_name} ({kernel}) to {output_path_png} and {output_path_pdf}")
            
            # Now create grid layout (2x2, 2x3, 3x3, etc.)
            import math
            n_cols = math.ceil(math.sqrt(n_models))
            n_rows = math.ceil(n_models / n_cols)
            fig_grid, axes_grid = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            if n_models == 1:
                axes_grid = [axes_grid]
            else:
                axes_grid = axes_grid.ravel()
            
            im_grid = None  # Store the last image for grid colorbar

            for idx, model in enumerate(selected_models):
                # Check if this is a baseline model
                if model in ['KNN', 'Decision Tree (gini)', 'Decision Tree (entropy)']:
                    if not baseline_models_data:
                        axes_grid[idx].set_visible(False)
                        continue
                    # Use baseline data
                    baseline_combined = pd.concat(baseline_models_data, ignore_index=True)
                    model_data = baseline_combined[baseline_combined['Model'] == model]
                else:
                    # Use SVM data
                    model_data = best_per_combo[(best_per_combo['Model'] == model) & 
                                            (best_per_combo['Kernel'] == kernel)]
                
                # Create pivot table
                pivot = model_data.pivot_table(
                    values='Test Acc',
                    index='Label_Noise',
                    columns='Feature_Noise',
                    aggfunc='mean'
                )

                # Reorder columns and index
                feature_order = ['Clean', '5%', '10%', '15%', '20%']
                label_order = ['0%', '5%', '10%', '15%', '20%']
                pivot = pivot.reindex(index=label_order, columns=feature_order)

                # Plot heatmap
                ax = axes_grid[idx]
                im_grid = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)

                # Set ticks
                ax.set_xticks(range(len(feature_order)))
                ax.set_yticks(range(len(label_order)))
                ax.set_xticklabels(feature_order)
                ax.set_yticklabels(['Clean' if label == '0%' else label for label in label_order])

                # Add text annotations
                for i in range(len(label_order)):
                    for j in range(len(feature_order)):
                        if not np.isnan(pivot.values[i, j]):
                            value = pivot.values[i, j]
                            text = ax.text(j, i, f'{value*100:.1f}%',
                                        ha='center', va='center', color='black', fontsize=14)

                # if model == "SKiP-average":
                #     display_model = "SKiP"
                if model == "SKiP-multiply-minmax":
                    display_model = "SKiP-minmax-multiply"
                elif model == "SKiP-average-minmax":
                    display_model = "SKiP-minmax-average"
                else:
                    display_model = model

                ax.set_title(f'{display_model}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Feature Outlier', fontsize=10)
                ax.set_ylabel('Label Outlier', fontsize=10)

            # Hide unused subplots in grid layout
            for idx in range(n_models, len(axes_grid)):
                axes_grid[idx].set_visible(False)
            
            # Add colorbar to grid layout
            if im_grid is not None:
                cbar_ax_grid = fig_grid.add_axes([0.92, 0.06, 0.03, 0.855])  
                cbar_grid = fig_grid.colorbar(im_grid, cax=cbar_ax_grid, orientation='vertical', label='Test Accuracy')
                cbar_grid.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))

            fig_grid.tight_layout(rect=[0, 0, 0.91, 0.95])
            output_path_png_grid = os.path.join(heatmap_dir, f'noise_heatmap_{dataset_name}_{kernel}_grid.png')
            output_path_pdf_grid = os.path.join(heatmap_pdf_dir, f'noise_heatmap_{dataset_name}_{kernel}_grid.pdf')
            fig_grid.savefig(output_path_png_grid, dpi=300, bbox_inches='tight')
            fig_grid.savefig(output_path_pdf_grid, bbox_inches='tight')
            plt.close(fig_grid)  # Close figure to free memory
            print(f"Saved heatmap (grid) for {dataset_name} ({kernel}) to {output_path_png_grid} and {output_path_pdf_grid}")

    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETED")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create noise heatmaps for selected models')
    parser.add_argument('--models', nargs='+', default=['SKiP-multiply', 'SKiP-multiply-minmax', 'SKiP-average', 'SKiP-average-minmax'],
                        help='List of models to visualize (default: NaiveSVM ProbSVM KNNSVM SKiP-average)')
    
    args = parser.parse_args()
    create_heatmaps(args.models)
