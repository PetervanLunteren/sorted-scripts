# script to visualise validation results of final model
# Peter van Lunteren, 17 Jan 2024

# You need to provide the path to the spp_plan.xlsx with an updated 'pivot_table'
# sheet (make sure the pivot table starts on row 1). Make sure the classes are 
# all there and are updated. 

# import packages
if(!require(tidyverse)){install.packages("tidyverse")};library(tidyverse)
if(!require(ggpubr)){install.packages("ggpubr")};library(ggpubr)

# clear env
rm(list = ls())

# user input
spp_plan = "/Users/peter/Documents/Addax/projects/2024-16-SCO/2024-16-SCO-spp-plan.xlsx"

# I'm not sure why - but the existing code did not work. Below is a minimal code to make some graphs. 
pt <- readxl::read_excel(spp_plan,
                         sheet = 'pivot_table',
                         skip = 1);str(pt)

# clean
colnames(pt)[1] = 'class'
pt = pt[pt$class != 'Grand Total',]
pt <- pt %>% rename(tot_n = `Grand Total`) %>%
  select(c(class, Local, 'Non-local', tot_n)) %>%
  filter(class != "(blank)")
pt$class <- tolower(pt$class)
pt[is.na(pt)] <- 0
pt <- pt %>%
  pivot_longer(cols = c("Local", "Non-local"), names_to = "Source", values_to = "Number of images")

# define function
plot_barchart <- function(plot_type = "all",
                          axis_range = NA,
                          title = NA,
                          subtitle = NA,
                          caption = NA){
  
  # filter
  if (plot_type == "all"){
    pt_sorted <- pt %>%
      arrange(tot_n)
  } else if (plot_type == "local-only") {
    pt_sorted <- pt %>%
      filter(Source == "Local") %>%
      arrange(`Number of images`)
  }
  
  # order
  unique_classes <- unique(pt_sorted$class)
  pt_sorted$Source <- factor(pt_sorted$Source, levels = c("Non-local", "Local"))
  
  # plot
  p <- ggplot(pt_sorted, aes(x = factor(class, levels = unique_classes), y = `Number of images`, fill = Source)) +
    geom_bar(stat = "identity", position = "stack") +
    theme_bw() +
    scale_fill_manual(values = c("Local" = "#1f77b4", "Non-local" = "#ff7f0e")) +
    coord_flip() +  
    labs(title = title) +
    theme(axis.title.y = element_blank(),)
  
  # adjust plot settings based on arguments
  if (plot_type == "all"){
    p <- p + geom_label(aes(class, tot_n, label=paste0(" ", tot_n)),
                        vjust = +0.5,
                        hjust = -0.2,
                        size = 3,
                        color = "black",
                        fill = "white")
  } else if (plot_type == "local-only") {
    p <- p + geom_label(aes(class, `Number of images`, label=paste0(" ", `Number of images`)),
                        vjust = +0.5,
                        hjust = -0.2,
                        size = 3,
                        color = "black",
                        fill = "white")
  }
  if (!is.na(subtitle)){
    p <- p + labs(subtitle = subtitle)
  }
  if (!is.na(caption)){
    p <- p +
      labs(caption = caption) +
      theme(plot.caption = element_text(face = "italic"))
  }
  if(!is.na(axis_range)){
    p <- p + coord_flip(ylim = c(0, axis_range))
  } else {
    max_y <- max(pt_sorted$`Number of images`) * 1.12
    p <- p + coord_flip(ylim = c(0, max_y))
  }
  
  # return plot
  return(p)
  
}

# overview
p_total <- plot_barchart(plot_type = 'all',
                            title = "Total number of training images per class",
                            subtitle = "First rule of thumb: a total of a few thousand images per class");p_total

# local only
p_local <- plot_barchart(plot_type = 'local-only',
                         title = "Number of local training images per class",
                         subtitle = "Second rule of thumb: a few hundred local images per class",
                         caption = "Local images are taken from the project area, whereas non-local\nimages are sourced from other comparable ecological studies.");p_local
# combine
p_combined <- ggarrange(p_total, p_local, ncol = 1, nrow = 2)

# save
ggsave(filename = paste0(dirname(spp_plan), "/training-data.png"),
       plot = p_combined,
       device = "png",
       width = 8,
       height = 10,
       dpi = 300)









# # zoomed in
# p_zoomed <- plot_barchart(plot_type = 'all',
#                           axis_range = 10000,
#                           title = "Total number of training images per class (zoomed in)",
#                           subtitle = "Rule of thumb: a total of a few thousand images per class");p_zoomed
# p_zoomed <- plot_barchart(plot_type = 'local-only',
#                           axis_range = 1000,
#                           title = "Total number of training images per class (zoomed in)",
#                           subtitle = "Rule of thumb: a few hundred local images per class");p_zoomed

######################## ORIGINAL CODE

# THIS code will also create a plot with n_imgs and accuracy. Not usually required.

# you'll need to have a local version of the final model dir (e.g., train40),
# including the validation xlsx files that are produced by 
# val-oosd-set.py and val-test-set.py. If you validated oosd on a previous model,
# make sure you replace the val-oosd-set-results.txt from the approprate model dir. 

train_dir = "/Users/peter/Documents/Addax/projects/2024-06-NZF/final_model/train13"

# import data
os <- readxl::read_excel(paste0(train_dir, "/out-of-of-sample-results.xlsx"),
                         col_names = c("class", "os_precision", "os_recall", "os_f1score", "os_support"),
                         skip = 1);str(os)
ts <- readxl::read_excel(paste0(train_dir, "/test-set-results.xlsx"),
                         col_names = c("class", "ts_precision", "ts_recall", "ts_f1score", "ts_support"),
                         skip = 1);str(ts)
pt <- readxl::read_excel(spp_plan,
                         sheet = 'pivot_table',
                         skip = 1);str(pt)

# clean
colnames(pt)[1] = 'class'
pt = pt[pt$class != 'Grand Total',]               # remove total row
pt <- pt %>% rename(tot_n = `Grand Total`) #,
                    # tot_local = `Local`,
                    # tot_non_local = `Non-local`)  # rename columns
# pt <- pt[, !colnames(pt) %in% c("Grand Total")]   
pt$class <- tolower(pt$class)                     # to lower case

# print non-class acuuracy values
print("OUT OF SAMPLE MAIN METRICS")
dplyr::filter(os, class %in% c("accuracy", "macro avg", "weighted avg"))[1:4]
print("TEST SET MAIN METRICS")
dplyr::filter(ts, class %in% c("accuracy", "macro avg", "weighted avg"))[1:4]

# merge
val_df <- full_join(os, ts, by = 'class')         # merge val dfs
val_df = val_df[val_df$class != 'accuracy',]      # remove non-class rows
val_df = val_df[val_df$class != 'macro avg',]     # remove non-class rows
val_df = val_df[val_df$class != 'weighted avg',]  # remove non-class rows
val_df <- full_join(val_df, pt, by = 'class')     # add img counts

# plot training images
options(scipen=4)
val_df <- val_df[order(val_df$tot_n),]
val_df$class <- factor(val_df$class, levels = val_df$class)
bar_df <- val_df[, c("class", "Local", "Non-local", "tot_n")] %>%
  pivot_longer(c("Local", "Non-local"), names_to = "Source", values_to = "Number of images")
bar_df[is.na(bar_df)] <- 0
plt <- 
  ggplot(bar_df, aes(x=class, y=`Number of images`, fill=Source)) +
  geom_bar(stat="identity", position = "stack") +
  geom_text(aes(class, tot_n, label=tot_n),
            vjust = +0.5,
            hjust = -0.2,
            size = 3,
            color = "darkgrey") +
  theme_bw() +
  # coord_flip(ylim = c(0, 70000)) +
  coord_flip(ylim = c(0, 1000000))+   # here you can set the axis range
  theme(
    axis.title.y = element_blank()
  );plot

# if you're happy with it, save it 
png(paste0(train_dir, "/trainset_counts.png"),
    width=20,
    height=12,
    units="cm",
    res=1200)
plt
dev.off()

# plot accuracy vs tot_n
val_df
plt <- 
  ggplot(val_df, aes(x=tot_n, y=ts_f1score)) +
  geom_point(shape = 1) +
  stat_smooth(aes(x=tot_n, y=ts_f1score), method = "loess", se = TRUE, level = 0.25, n = 1000) +
  coord_cartesian(xlim = c(0, 50000)) +
  # coord_cartesian() +
  # coord_cartesian(xlim = c(0, 7100)) +
  # geom_hline(yintercept=0.9, linetype="dashed", color = "red") + 
  # geom_vline(xintercept=7000, linetype="dashed", color = "red") + 
  ylim(0.93, 1) +
  theme_bw() +
  labs(y = "Accuracy",
       x = "Number of images");plt

# if you're happy with it, save it 
png(paste0(train_dir, "/acc_trainset.png"),
    width=20,
    height=12,
    units="cm",
    res=1200)
plt
dev.off()



