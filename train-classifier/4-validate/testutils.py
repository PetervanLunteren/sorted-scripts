# a bunch of functions that are useful for testing and visualising

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import yaml
from tqdm import trange
import pandas as pd
import torch
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF # pip install fpdf
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import random
from matplotlib.gridspec import GridSpec
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import torch.nn.functional as F

# color settings
title_color_1 = (10, 40, 80)
title_color_2 = (30, 80, 160)
title_color_3 = (200, 225, 250)
title_color_4 = (220, 235, 252)
title_color_5 = (240, 245, 255)

# table and header settings
header_width = 180 # this is the width of the header
part = header_width / 6  # given that the table always has 5 cols -> 2 parts for first column, 1 each for the rest
first_col_width = part * 2
remaining_col_width = part
font_size = 7 
row_height = 5
char_cutoff_length_table = 50
char_cutoff_length_conf_matr = 11


def plot_img_availability_vs_accuracy(classification_report_dict, max_plateau, max_growth_rate, dst_dir, spp_plan_dict):

    # get the data
    availability_data = {k: v for k, v in classification_report_dict.items() if k not in ["accuracy", "macro avg", "weighted avg"]}

    # remove nan values
    for cat, metrics in availability_data.items():
        if np.isnan(metrics['f1-score']):
            metrics['f1-score'] = 0.0
            
    # get values
    f1_scores = np.array([info['f1-score'] for info in availability_data.values()])
    
    # adjust the keys of spp_plan_dict to all lower case, so they match
    spp_plan_dict = {k.lower(): v for k, v in spp_plan_dict.items()}

    # Get total_images from spp_plan_dict as support values, so we use the actual img counts, not the ones that we downsampled
    supports = np.array([
        spp_plan_dict.get(cat.lower(), {}).get("total_images", 0)  # Default to 0 if category not found
        for cat in availability_data.keys()
    ])
    categories = [key for key in availability_data.keys()]

    # Normalize the support values to [0, 1]
    supports_normalized = supports / max(supports)

    # Define the Saturated Exponential Model
    def exp_saturation(x, L, k):
        return L * (1 - np.exp(-k * x))

    # Fit the model 
    popt, pcov = curve_fit(
        exp_saturation,
        supports_normalized,
        f1_scores,
        p0=[0.92,  # initial guess for plateau value (L)
            1],    # initial guess for growth rate (k)
        bounds=([0,     # constrain plateau (L) lower bound
                0],    # constrain growth rate (k) lower bound
                [max_plateau,  # constrain plateau (L) upper bound
                max_growth_rate])   # constrain growth rate (k) upper bound
    )

    L_opt, k_opt = popt
    x_fine = np.linspace(0, 1, 500)
    y_fine = exp_saturation(x_fine, L_opt, k_opt)

    # with fit
    plt.figure(figsize=(10, 6))
    plt.scatter(supports, f1_scores, color='b', alpha=0.7, label='Data')
    for i, category in enumerate(categories):
        plt.text(supports[i], f1_scores[i], f"  {category}", fontsize=9, ha='left', va='center')
    plt.plot(x_fine * max(supports), y_fine, label=f'Fitted exponential model (L={round(L_opt, 2)}, k = {round(k_opt, 1)})', color='red', lw=2, alpha=0.3, zorder=10)
    # plt.title('Image availbility vs class accuracy')
    plt.xlabel('Image availbility')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    # plt.xlim(0, 100000)  # Limit the x-axis
    plt.legend()
    plt.grid(alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    path = os.path.join(dst_dir, 'image_availability_vs_accuracy_with_fit.png')
    plt.savefig(path)
    plt.close()
    print(f"Created: .\\image_availability_vs_accuracy_with_fit.png")

    # # without fit
    # plt.figure(figsize=(10, 6))
    # plt.scatter(supports, f1_scores, color='b', alpha=0.7, label='Data')
    # for i, category in enumerate(categories):
    #     plt.text(supports[i], f1_scores[i], f"  {category}", fontsize=9, ha='left', va='center')
    # plt.title('Image availbility vs class accuracy')
    # plt.xlabel('Image availbility')
    # plt.ylabel('Accuracy')
    # plt.ylim(0, 1)
    # plt.grid(alpha=0.3)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # path = os.path.join(dst_dir, 'image_availability_vs_accuracy_without_fit.png')
    # plt.savefig(path)
    # plt.close()
    # print(f"Created: .\\image_availability_vs_accuracy_without_fit.png")
    
    return path

def plot_image_availability_bar_chart(data, dst_dir):

    # adjust the keys of data to all lower case, so they match
    data = {k.lower(): v for k, v in data.items()}

    # Sort the data by total image count, from large to small
    sorted_data = {key: value for key, value in sorted(data.items(), key=lambda item: item[1]["total_images"], reverse=False)}

    categories = list(sorted_data.keys())
    local_counts = np.array([sorted_data[key]["local_images"] for key in categories])
    non_local_counts = np.array([sorted_data[key]["non_local_images"] for key in categories])
    total_counts = np.array([sorted_data[key]["total_images"] for key in categories])

    y_pos = np.arange(len(categories))

    figsize = (9, min(3 + len(data.items()) * 0.15, 12)) # figsize based on the len classes
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y_pos, local_counts, color='royalblue', label='Local Images')
    ax.barh(y_pos, non_local_counts, left=local_counts, color='lightcoral', label='Non-Local Images')

    # Annotate total count at the end of each bar
    for i, total in enumerate(total_counts):
        ax.text(total + 500, y_pos[i], f'{total}', va='center', fontsize=11)

    # Add margin to the x-axis for the largest bar
    ax.set_xlim(0, total_counts.max() * 1.1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Images")
    # ax.set_title("Image avilability by class")

    # Place legend in the lower right corner
    ax.legend(loc='lower right')

    plt.tight_layout()
    # plt.show()
    
    img_availability_barchart_fpath = os.path.join(dst_dir, 'image_availability_barchart.png')
    plt.savefig(img_availability_barchart_fpath)
    plt.close()
    print(f"Created: .\\image_availability_barchart.png")
    
    # return filename
    return img_availability_barchart_fpath



def plot_confusion_matrices(cm, classes, dst_dir, type):

    tick_marks = np.arange(len(classes))
    short_classes = ["..." + label[-(char_cutoff_length_conf_matr-1):] if len(label) > char_cutoff_length_conf_matr else label for label in classes]
    
    # Plot confusion matrix
    if type == "non-normalized":
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.title('Confusion matrix (non-normalized)')
        plt.colorbar()
        plt.xticks(tick_marks, short_classes, rotation=90)
        plt.yticks(tick_marks, short_classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        cm_path = os.path.join(dst_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"Created: .\\confusion_matrix.png")
        return cm_path

    # Plot normalized confusion matrix
    elif type == "normalized":
        row_sums = cm.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Prevent division by zero
        cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.title('Confusion matrix (normalized)')
        plt.colorbar()
        plt.xticks(tick_marks, short_classes, rotation=90)
        plt.yticks(tick_marks, short_classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        # Annotate the matrix (only if less than 50 classes - otherwise too little space)
        if len(classes) < 50:
            threshold = cm_normalized.max() / 2.
            for i, j in np.ndindex(cm_normalized.shape):
                plt.text(j, i, f"{cm_normalized[i, j]:.2f}" if cm_normalized[i, j] > 0.005 else "",
                        horizontalalignment="center",
                        color="white" if cm_normalized[i, j] > threshold else "black")
            
        norm_cm_path = os.path.join(dst_dir, 'confusion_matrix_normalized.png')
        plt.savefig(norm_cm_path)
        plt.close()
        print(f"Created: .\\confusion_matrix_normalized.png")
        return norm_cm_path
    
# Function to calculate the aspect ratio
def get_aspect_ratio(image_path):
    with Image.open(image_path) as img:
        width, height = img.size  # Get image dimensions
        aspect_ratio = width / height  # Calculate the aspect ratio
    return aspect_ratio

def plot_images(df, true_class, dst_dir, mode, num_images=21):
    """
    Plots images of incorrect predictions for a given true class.

    Args:
    - df (pd.DataFrame): DataFrame containing columns ["FilePath", "PredictedLabel", "Confidence", "corr", "true"].
    - true_class (str): The true class to filter.
    - num_images (int): The number of images to display.
    """
        
    # Filter data for the true class
    if mode == "false class":
        chosen_preds = df[~df["corr"] & (df["true"] == true_class)].sort_values(by="conf", ascending=False).head(num_images)
    elif mode == "false other":
        chosen_preds = df[~df["corr"] & (df["pred"] == true_class)].sort_values(by="conf", ascending=False).head(num_images)

    # Ensure grid has exactly `num_images` slots
    if len(chosen_preds) < num_images:
        missing_rows = num_images - len(chosen_preds)
        empty_rows = pd.DataFrame({col: [None] * missing_rows for col in chosen_preds.columns})
        chosen_preds = pd.concat([chosen_preds, empty_rows], ignore_index=True)

    # Setup figure and grid layout
    img_cols = 7  # Fixed columns
    img_rows = num_images // img_cols + (1 if num_images % img_cols != 0 else 0)  # Calculate rows for 6 columns

    fig = plt.figure(figsize=(img_cols * 1.7, img_rows * 1.7))  # Reduce the width per column
    gs = GridSpec(img_rows, img_cols, figure=fig, wspace=0.2, hspace=0.5)  # Adjust wspace for horizontal spacing

    # Plot the images
    for idx, row in enumerate(chosen_preds.itertuples(index=False)):
        row_idx, col_idx = divmod(idx, img_cols)  # Grid position
        ax = fig.add_subplot(gs[row_idx, col_idx])

        # Plot the image or placeholder
        img_path = row.path
        if img_path and isinstance(img_path, str):
            try:
                img = Image.open(img_path).resize((182, 182), Image.BICUBIC)
                ax.imshow(img)
                true_title = row.true if len(row.true) < 13 else row.true[0:13] + ".."
                pred_title = row.pred if len(row.pred) < 10 else row.pred[0:10] + ".."
                ax.set_title(f"T: {true_title}\nP: {pred_title} {min(round(row.conf * 100, 1), 99.9)}%", fontsize=8)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                ax.axis('off')
        else:
            ax.imshow(np.ones((120, 120, 3)) * 0.5)  # Grey placeholder
            ax.axis('off')

        ax.axis('off')

    # Adjust subplot layout
    fig.subplots_adjust(top=0.85, wspace=0.2, hspace=0.4, bottom = 0.01)  # Top adjusts room for the suptitle
    

    filename = os.path.join(dst_dir, f"images-{true_class.replace(' ', '-')}-{mode.replace(' ', '-')}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # close the figure
    plt.close(fig)
    
    # return filename
    return filename


def plot_class_counts(df, true_class, dst_dir, mode):
    # # Filter the DataFrame for incorrect predictions with the true class
    if mode == "false class":
        chosen_preds = df[~df["corr"] & (df["true"] == true_class)]
        class_counts = chosen_preds["pred"].value_counts()
    elif mode == "false other":
        chosen_preds = df[~df["corr"] & (df["pred"] == true_class)]
        class_counts = chosen_preds["true"].value_counts()

    # Create the figure and GridSpec for subplots
    fig = plt.figure(figsize=(10, 3))
    gs = GridSpec(1, 1)  # You can adjust the layout of subplots as needed
    
    # Create the bar chart with sorted data
    sorted_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))
    bar_ax = fig.add_subplot(gs[0])
    bars = bar_ax.bar(sorted_counts.keys(), sorted_counts.values())
    bar_ax.set_ylabel("Count", fontsize=10)

    # Rotate the x-axis labels by 45 degrees and move them a bit to the left
    bar_ax.set_xticks(range(len(sorted_counts)))  # Set x-ticks based on the number of bars
    bar_ax.set_xticklabels(sorted_counts.keys(), rotation=20, ha="right", rotation_mode="anchor")  # Rotate and move left

    # Adjust the plot's layout to give extra space for the tilted labels
    fig.tight_layout(pad=2.0)  # Increase the padding between the plot and the figure's edge

    # Remove the outer spines (top, right, left, bottom) to avoid the border
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)
    bar_ax.spines['left'].set_visible(True)
    bar_ax.spines['bottom'].set_visible(True)

    # Add percentage labels on top of the bars
    total_count = len(chosen_preds)
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_count) * 100
        bar_ax.text(bar.get_x() + bar.get_width() / 2, height + 0.2, f'{percentage:.2f}%', 
                    ha='center', va='bottom', fontsize=10)

    filename = os.path.join(dst_dir, f"counts-{true_class.replace(' ', '-')}-{mode.replace(' ', '-')}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # close the figure
    plt.close(fig)
    
    # return fig
    return filename

def plot_confidence_distribution(df, dst_dir, mode, true_class = None):
    # Confidence distribution plot (using counts)
    if mode == "false class":
        correct_conf = df[df["corr"] & (df["true"] == true_class)]["conf"]
        incorrect_conf = df[~df["corr"] & (df["true"] == true_class)]["conf"]
    elif mode == "false other":
        correct_conf = df[df["corr"] & (df["pred"] == true_class)]["conf"]
        incorrect_conf = df[~df["corr"] & (df["pred"] == true_class)]["conf"]
    elif mode == "all":
        correct_conf = df[df["corr"]]["conf"]
        incorrect_conf = df[~df["corr"]]["conf"]

    # Create the figure and GridSpec for subplots
    if mode != "all":
        fig = plt.figure(figsize=(10, 3))
    else:
        fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 2, width_ratios=[1, 1])  # Two subplots with equal width
    bar_ax = fig.add_subplot(gs[0])  # Left: Bar chart
    density_ax = fig.add_subplot(gs[1])  # Right: Density plot

    # Bar chart (left)
    sns.histplot(correct_conf, ax=bar_ax, color='green', kde=False, stat='count', bins=20, alpha=0.5, linewidth=0, label='Correct')
    sns.histplot(incorrect_conf, ax=bar_ax, color='red', kde=False, stat='count', bins=20, alpha=0.5, linewidth=0, label='Incorrect')
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)
    bar_ax.set_xlabel("Confidence", fontsize=12)
    bar_ax.set_ylabel("Count", fontsize=12)
    bar_ax.legend(["Correct", "Incorrect"], loc="upper left")
    bar_ax.set_title("absolute", fontsize=12, weight='bold')

    # Density plot (right)
    sns.kdeplot(correct_conf, ax=density_ax, color='green', fill=True, alpha=0.5, linewidth=1.5, label='Correct', warn_singular=False)
    sns.kdeplot(incorrect_conf, ax=density_ax, color='red', fill=True, alpha=0.5, linewidth=1.5, label='Incorrect', warn_singular=False)
    density_ax.spines['top'].set_visible(False)
    density_ax.spines['right'].set_visible(False)
    density_ax.set_xlabel("Confidence", fontsize=12)
    density_ax.set_ylabel("Density", fontsize=12)
    density_ax.legend(loc="upper left")
    density_ax.set_title("relative", fontsize=12, weight='bold')

    # Adjust layout
    fig.tight_layout(pad=0.2, rect=[0, 0, 1, 0.93])  # Adjust space for the suptitle
    if mode != "all":
        filename = os.path.join(dst_dir, f"confchart-{true_class.replace(' ', '-')}-{mode.replace(' ', '-')}.png")
    elif mode == "all":
        filename = os.path.join(dst_dir, f"confchart-all.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # close the figure
    plt.close(fig)
    
    # return fig
    return filename

# Initialize FPDF object
class PDF(FPDF):
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'C')
        self.ln(5)

def create_front_page(df, dst_dir, classification_report_dict, spp_plan_dict, split, split_count_df, location_counts, confusion_matrix, all_classes, project_name, column_name_to_mapping):

    img_availability_barchart_fpath = plot_image_availability_bar_chart(spp_plan_dict, dst_dir)

    # Create PDF instance
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_draw_color(*title_color_3)
    pdf.set_line_width(1)

    # Add the main title
    pdf.set_fill_color(*title_color_1)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', size=20)
    pdf.cell(0, 10, border=0, fill=True, ln=True, align='C',
            txt=f"Model evaluation report for {project_name}")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(0)

    # Add the taxon level
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.set_fill_color(*title_color_4)
    pdf.set_font("Arial", '', size=12)
    if column_name_to_mapping.startswith("level_"):
        subtitle = f"Predictions aggregated to the taxonomic level '{column_name_to_mapping.replace('level_', '')}'"
    elif column_name_to_mapping.startswith("only_above_"):
        subtitle = f"Predictions aggregated to only categories with more than {int(column_name_to_mapping.replace('only_above_', '')):,} training samples"
    elif column_name_to_mapping == "model_class":
        subtitle = f"Based on the true model classes (no prediction aggregations)"
    pdf.cell(header_width, 6, border=0, fill=True, ln=True, align='C', txt=f"{subtitle}")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # set table vars
    pdf.set_line_width(0.2)
    pdf.set_draw_color(*title_color_3)
    
    # Add the subtitle title to the page
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.set_fill_color(*title_color_2)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', size=15)
    pdf.cell(header_width, 10, border=0, fill=True, ln=True, align='',
            txt=f"1 - Dataset and split information")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Add table header for location counts **AFTER** the last plot
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.set_fill_color(*title_color_3)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(header_width, 8, ln=True, align='', fill=True,
            txt=f"1.1 - Location subsets")

    # Add headers for location counts
    pdf.set_font('Arial', 'B', font_size)
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.cell(first_col_width, row_height, '', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, 'Total', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, 'Train', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, 'Val', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, 'Test', border=1, align='C')
    pdf.ln()
    
    # Add values for location counts
    pdf.set_font('Arial', '', font_size)
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.cell(first_col_width, row_height, 'Locations', border=1, align='R')
    pdf.cell(remaining_col_width, row_height, f'{location_counts["number of total locations"]}', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, f'{location_counts["number of train locations"]}', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, f'{location_counts["number of val locations"]}', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, f'{location_counts["number of test locations"]}', border=1, align='C')
    pdf.ln()
    pdf.ln(5)

    # Add table header for data split
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.set_fill_color(*title_color_3)
    pdf.set_font("Arial", 'B', size=12)
    page_width = pdf.w
    x_position = (page_width - header_width) / 2 
    pdf.set_x(x_position)
    pdf.cell(header_width, 8, ln=True, align='', fill=True,
            txt=f"1.2 - Image subsets")

    # Add headers for data split
    pdf.set_font('Arial', 'B', font_size)
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.cell(first_col_width, row_height, '', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, 'Total', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, 'Train', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, 'Val', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, 'Test', border=1, align='C')
    pdf.ln()

    # Add rows for data split
    pdf.set_font('Arial', '', font_size)
    for key, value in split_count_df.iterrows():
        pdf.set_x((pdf.w - header_width) / 2)

        if len(key) > char_cutoff_length_table:
            class_title = "..." + key[-(char_cutoff_length_table-1):]
        else:
            class_title = key
        pdf.cell(first_col_width, row_height, class_title + " ", border=1, align='R')

        # class_title = key if len(key) < 16 else key[0:16] + ".."
        # class_title = class_title.capitalize()
        for val in value:
            pdf.cell(remaining_col_width, row_height, val, border=1, align='C')
        pdf.ln()
    pdf.ln()

    # Add image availability by class
    pdf.add_page()
    plot_y_margin = 10
    plot_x_margin = 15
    plot_curr_y = 15  # Starting Y position
    page_width = pdf.w - 2 * plot_x_margin

    for i, (plot, title) in enumerate([
        [img_availability_barchart_fpath, "Image availability by class"]
    ]):
        # Define a fixed width for the header and plot
        page_width = pdf.w
        x_position = (page_width - header_width) / 2  # Centering X position

        # Determine the Y-position for the plot
        plot_y = plot_curr_y + plot_y_margin  
        header_y = plot_y - 8  # Position header just above the plot

        # Set header position and add title
        pdf.set_fill_color(*title_color_3)
        pdf.set_font("Arial", 'B', size=12)
        pdf.set_xy(x_position, header_y)
        pdf.cell(header_width, 8, ln=True, align='', fill=True, txt=f"1.3 - {title}")

        # Add the plot (same width as header)
        aspect_ratio = get_aspect_ratio(plot)
        plot_width = header_width  # Make plot width same as header width
        plot_height = plot_width / aspect_ratio  # Maintain aspect ratio
        pdf.image(plot, x=x_position, y=plot_y, w=plot_width)  
        pdf.rect(x_position, plot_y, plot_width, plot_height, style='D')  # Add border

        # Update cursor for the next section
        plot_curr_y = plot_y + plot_height + 4  # Ensure spacing after plot

    # Save the PDF
    output_pdf = os.path.join(dst_dir, f"main_front_page.pdf")
    pdf.output(output_pdf)

    # Clean up temporary files
    for plot in [img_availability_barchart_fpath]:
        os.remove(plot)
        
    print(f"Created: .\\main_front_page.pdf")
    
    return output_pdf


def create_split_front_page(df, dst_dir, classification_report_dict, spp_plan_dict, split, split_count_df, location_counts, confusion_matrix, all_classes, project_name, paragraph_idx):

    all_class_counts_fpath = plot_confidence_distribution(df, dst_dir, mode= "all")
    img_availability_vs_accuracy_fpath = plot_img_availability_vs_accuracy(classification_report_dict = classification_report_dict,
                                                                            max_plateau = 0.97,            # HWI = 0.91, OHI = 1,  CAN = 0.97
                                                                            max_growth_rate = 600,         # HWI = 29.4, OHI = 50, CAN = 600
                                                                            dst_dir = dst_dir,
                                                                            spp_plan_dict = spp_plan_dict)
    cm_fpath = plot_confusion_matrices(confusion_matrix, all_classes, dst_dir, "non-normalized")
    norm_cm_fpath = plot_confusion_matrices(confusion_matrix, all_classes, dst_dir, "normalized")

    # Create PDF instance
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_draw_color(*title_color_3)
    pdf.set_line_width(1)
    split_title = "validation" if split == "val" else "test"

    # set table vars
    pdf.set_line_width(0.2)
    pdf.set_draw_color(*title_color_3)

    # Add the subtitle title to the page
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.set_fill_color(*title_color_2)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', size=15)
    pdf.cell(header_width, 10, border=0, fill=True, ln=True, align='',
            txt=f"{paragraph_idx} - Metrics based on the {split_title} subset")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Add table header for confusion matrix
    pdf.set_fill_color(*title_color_3)
    pdf.set_font("Arial", 'B', size=12)
    page_width = pdf.w
    x_position = (page_width - header_width) / 2 
    pdf.set_x(x_position)  # Set Y-position after the last plot
    pdf.cell(header_width, 8, ln=True, align='', fill=True,
            txt=f"{paragraph_idx}.1 - Classification report")
    
    # Add headers for confusion matrix
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.set_font('Arial', 'B', font_size)
    pdf.cell(first_col_width, row_height, '', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, 'Precision', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, 'Recall', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, 'F1-Score', border=1, align='C')
    pdf.cell(remaining_col_width, row_height, 'Support', border=1, align='C')
    pdf.ln()
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.set_font('Arial', '', font_size)
    
    # add rows for confusion matrix
    for key, value in classification_report_dict.items():
        pdf.set_x((pdf.w - header_width) / 2)

        # class_title = key if len(key) < 16 else key[0:16] + ".."

        if len(key) > char_cutoff_length_table:
            class_title = "..." + key[-(char_cutoff_length_table-1):]
        else:
            class_title = key

        class_title = class_title.capitalize()
        if isinstance(value, dict):
            pdf.cell(first_col_width, row_height, class_title + " ", border=1, align='R')
            pdf.cell(remaining_col_width, row_height, f"{round(value['precision']*100, 1)}%", border=1, align='C')
            pdf.cell(remaining_col_width, row_height, f"{round(value['recall']*100, 1)}%", border=1, align='C')
            pdf.cell(remaining_col_width, row_height, f"{round(value['f1-score']*100, 1)}%", border=1, align='C')
            pdf.cell(remaining_col_width, row_height, f"{int(value['support'])}", border=1, align='C')
            pdf.ln()
        else:
            pdf.cell(first_col_width, 1, '', border=1)
            pdf.cell(remaining_col_width, 1, '', border=1)
            pdf.cell(remaining_col_width, 1, '', border=1) 
            pdf.cell(remaining_col_width, 1, '', border=1)
            pdf.cell(remaining_col_width, 1, '', border=1)
            pdf.ln()
            pdf.set_x((pdf.w - header_width) / 2)
            pdf.cell(first_col_width, row_height, key.capitalize(), border=1, align='R')
            pdf.cell(remaining_col_width, row_height, '', border=1)
            pdf.cell(remaining_col_width, row_height, '', border=1)
            pdf.cell(remaining_col_width, row_height, f"{round(value*100, 1)}%" if isinstance(value, float) else str(value), border=1, align='C')
            pdf.cell(remaining_col_width, row_height, '', border=1)
            pdf.ln()

    # Add plots to the PDF
    pdf.add_page()
    plot_y_margin = 10
    plot_x_margin = 15
    plot_curr_y = 15
    page_width = pdf.w - 2 * plot_x_margin

    for i, (plot, title) in enumerate([
        [all_class_counts_fpath, "Confidence distribution"], 
        [img_availability_vs_accuracy_fpath, "Image availability vs accuracy"]
    ]):
        # Define a fixed width for the header and plot
        page_width = pdf.w
        x_position = (page_width - header_width) / 2  # Centering X position

        # Determine the Y-position for the plot
        plot_y = plot_curr_y + plot_y_margin  
        header_y = plot_y - 8  # Position header just above the plot

        # Set header position and add title
        pdf.set_fill_color(*title_color_3)
        pdf.set_font("Arial", 'B', size=12)
        pdf.set_xy(x_position, header_y)
        pdf.cell(header_width, 8, ln=True, align='', fill=True, txt=f"{paragraph_idx}.{i+2} - {title}")

        # Add the plot (same width as header)
        aspect_ratio = get_aspect_ratio(plot)
        plot_width = header_width  # Make plot width same as header width
        plot_height = plot_width / aspect_ratio  # Maintain aspect ratio
        pdf.image(plot, x=x_position, y=plot_y, w=plot_width)  
        pdf.rect(x_position, plot_y, plot_width, plot_height, style='D')  # Add border

        # Update cursor for the next plot
        plot_curr_y = plot_y + plot_height + 5  # Space between plots

    # new page for the confusion matrices
    pdf.add_page()
    plot_y_margin = 10 
    plot_x_margin = 15
    plot_curr_y = 15
    page_width = pdf.w - 2 * plot_x_margin

    for i, (plot, title) in enumerate([
        [cm_fpath, "Confusion matrix (non-normalized)"], 
        [norm_cm_fpath, "Confusion matrix (normalized)"]
    ]):
        # Define a fixed width for the header and plot
        page_width = pdf.w
        x_position = (page_width - header_width) / 2  # Centering X position

        # Determine the Y-position for the plot
        plot_y = plot_curr_y + plot_y_margin  
        header_y = plot_y - 8  # Position header just above the plot

        # Set header position and add title
        pdf.set_fill_color(*title_color_3)
        pdf.set_font("Arial", 'B', size=12)
        pdf.set_xy(x_position, header_y)
        pdf.cell(header_width, 8, ln=True, align='', fill=True, txt=f"{paragraph_idx}.{i+4} - {title}")

        # Add the plot (same width as header)
        aspect_ratio = get_aspect_ratio(plot)
        
        # Scale the width to reduce height (e.g., 80% of the original width)
        scaled_width = header_width * 0.8  # Adjust the factor here (0.8 means 80% width)
        plot_height = scaled_width / aspect_ratio  # Maintain aspect ratio

        # Center the plot horizontally
        centered_x_position = (page_width - scaled_width) / 2  # Calculate center position for plot

        pdf.image(plot, x=centered_x_position, y=plot_y, w=scaled_width)  
        pdf.rect(centered_x_position, plot_y, scaled_width, plot_height, style='D')  # Add border

        # Update cursor for the next plot
        plot_curr_y = plot_y + plot_height + 5  # Space between plots

    # Save the PDF
    output_pdf = os.path.join(dst_dir, f"{split}_front_page.pdf")
    pdf.output(output_pdf)

    # Clean up temporary files
    for plot in [all_class_counts_fpath,
                 img_availability_vs_accuracy_fpath,
                 cm_fpath,
                 norm_cm_fpath]:
        os.remove(plot)
        
    print(f"Created: .\\{split}_front_page.pdf")

    return output_pdf


def create_class_specific_error_report_PDF(df, chosen_class, dst_dir, metrics, paragraph_idx, idx):

    # page RECALL: How many of the true wild boars were classified as other animals?
    false_class_imgs_fpath = plot_images(df, chosen_class, dst_dir, "false class")
    false_class_counts_fpath = plot_class_counts(df, chosen_class, dst_dir, "false class")
    false_class_conf_distr_fpath = plot_confidence_distribution(df, dst_dir, "false class", true_class = chosen_class)

    # Page precision: How many other animals were classified as wild boars?
    false_other_fig_fpath = plot_images(df, chosen_class, dst_dir, "false other")
    false_other_counts_fpath = plot_class_counts(df, chosen_class, dst_dir, "false other")
    false_other_conf_distr_fpath = plot_confidence_distribution(df, dst_dir, "false other", true_class = chosen_class)

    # Create PDF instance
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    
    pdf.set_line_width(1)

    # Add the subtitle title to the page
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.set_fill_color(*title_color_3)
    pdf.set_font("Arial", 'B', size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(header_width, 8, border=0, fill=True, ln=True, align='',
            txt=f"{paragraph_idx}.{idx} - Class-specific error report for {chosen_class}")
    pdf.set_text_color(0, 0, 0)
    pdf.set_line_width(0.2)
    pdf.set_draw_color(*title_color_4)

    # Add table header for location counts **AFTER** the last plot
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.set_fill_color(*title_color_4)
    pdf.set_font("Arial", '', size=12)
    pdf.cell(header_width, 8, ln=True, align='', fill=True,
            txt=f"{paragraph_idx}.{idx}.1 - General statistics")

    # Add the formatted metrics string
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, ln=True, align='C',
            txt=f"Accuracy: {round(metrics['f1-score'] * 100, 1)}%  |  "
            f"Precision: {round(metrics['precision'] * 100, 1)}%  |  "
            f"Recall: {round(metrics['recall'] * 100, 1)}%  |  "
            f"Tested on {int(metrics['support'])} images\n")
    pdf.ln(2)

    # # Add table header for location counts **AFTER** the last plot
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.set_fill_color(*title_color_4)
    pdf.set_font("Arial", '', size=12)
    cut_off_class = chosen_class[:15] + "..." if len(chosen_class) > 15 else chosen_class
    pdf.cell(header_width, 8, ln=True, align='', fill=True,
            txt=f"{paragraph_idx}.{idx}.2 - Recall report (Which {cut_off_class} images were classified as other?)")
    
    # Add plots to the PDF
    plot_y_margin = 10
    plot_x_margin = 15
    plot_curr_y = 44
    page_width = pdf.w - 2 * plot_x_margin

    for i, (plot, title) in enumerate([
        [false_class_counts_fpath, "False negative label count"], 
        [false_class_imgs_fpath, "False negative images with descending confidence (T = true, P = predicted)"],
        [false_class_conf_distr_fpath, "False negative confidence distribution"],
    ]):
        # Define a fixed width for the header and plot
        # cell_width = 180  # Fixed width for both header and plot
        page_width = pdf.w
        x_position = (page_width - header_width) / 2  # Centering X position

        # Determine the Y-position for the plot
        plot_y = plot_curr_y + plot_y_margin  
        header_y = plot_y - 8  # Position header just above the plot

        # Set header position and add title
        pdf.set_fill_color(*title_color_5)
        pdf.set_font("Arial", '', size=10)
        pdf.set_xy(x_position, header_y)
        pdf.cell(header_width, 8, ln=True, align='', fill=True, txt=f"{paragraph_idx}.{idx}.2.{i+1} - {title}")

        # Add the plot (same width as header)
        aspect_ratio = get_aspect_ratio(plot)
        plot_width = header_width  # Make plot width same as header width
        plot_height = plot_width / aspect_ratio  # Maintain aspect ratio
        pdf.image(plot, x=x_position, y=plot_y, w=plot_width)  
        pdf.set_draw_color(*title_color_5)
        pdf.rect(x_position, plot_y, plot_width, plot_height, style='D')  # Add border

        # Update cursor for the next plot
        plot_curr_y = plot_y + plot_height + 5  # Space between plots

    ###### PAGE 2
    pdf.add_page()

    # header
    pdf.set_x((pdf.w - header_width) / 2)
    pdf.set_fill_color(*title_color_4)
    pdf.set_font("Arial", '', size=12)
    cut_off_class = chosen_class[:15] + "..." if len(chosen_class) > 15 else chosen_class
    pdf.cell(header_width, 8, ln=True, align='', fill=True,
            txt=f"{paragraph_idx}.{idx}.3 - Precision report (Which others were classified as {cut_off_class}?)")

    # Add plots to the PDF
    plot_y_margin = 10
    plot_x_margin = 15
    plot_curr_y = 16
    page_width = pdf.w - 2 * plot_x_margin

    for i, (plot, title) in enumerate([
        [false_other_counts_fpath, " False positive label count"], 
        [false_other_fig_fpath, " False positive images with descending confidence (T = true, P = predicted)"],
        [false_other_conf_distr_fpath, " False positive confidence distribution"],
    ]):
        # Define a fixed width for the header and plot
        page_width = pdf.w
        x_position = (page_width - header_width) / 2  # Centering X position

        # Determine the Y-position for the plot
        plot_y = plot_curr_y + plot_y_margin  
        header_y = plot_y - 8  # Position header just above the plot

        # Set header position and add title
        pdf.set_fill_color(*title_color_5)
        pdf.set_font("Arial", '', size=10)
        pdf.set_xy(x_position, header_y)
        pdf.cell(header_width, 8, ln=True, align='', fill=True, txt=f"{paragraph_idx}.{idx}.3.{i+1} - {title}")

        # Add the plot (same width as header)
        aspect_ratio = get_aspect_ratio(plot)
        plot_width = header_width  # Make plot width same as header width
        plot_height = plot_width / aspect_ratio  # Maintain aspect ratio
        pdf.image(plot, x=x_position, y=plot_y, w=plot_width)  
        pdf.set_draw_color(*title_color_5)
        pdf.rect(x_position, plot_y, plot_width, plot_height, style='D')  # Add border

        # Update cursor for the next plot
        plot_curr_y = plot_y + plot_height + 5  # Space between plots
        
    # Save the PDF
    output_pdf = os.path.join(dst_dir, f"error_report_{chosen_class.replace(' ', '_')}.pdf")
    pdf.output(output_pdf)

    # Clean up temporary files
    for plot in [false_class_imgs_fpath,
                 false_other_fig_fpath,
                 false_class_counts_fpath,
                 false_other_counts_fpath,
                 false_class_conf_distr_fpath,
                 false_other_conf_distr_fpath]:
        os.remove(plot)

    print(f"Created: .\\error_report_{chosen_class.replace(' ', '_')}.pdf")
    
    return output_pdf

