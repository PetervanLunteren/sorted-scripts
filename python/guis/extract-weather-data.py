import tkinter as tk
from tkinter import IntVar
from tkcalendar import DateEntry

def extract_data():
    selected_source = source_var.get()
    start_date = start_date_entry.get_date()
    end_date = end_date_entry.get_date()
    print("Extracting data from:", selected_source)
    print("Start Date:", start_date)
    print("End Date:", end_date)

def calculate_metrics():
    selected_metrics = []
    if precipitation_var.get():
        selected_metrics.append("Precipitation")
    if temperature_var.get():
        selected_metrics.append("Temperature")
    if sun_hours_var.get():
        selected_metrics.append("Sun Hours")
    print("Calculating metrics:", selected_metrics)

def plot_results():
    selected_plots = []
    if graph_var.get():
        selected_plots.append("Graph")
    if table_var.get():
        selected_plots.append("Table")
    if map_var.get():
        selected_plots.append("Map")
    print("Plotting results:", selected_plots)

# Create the main application window
root = tk.Tk()
root.title("Weather Data Analysis")

# Create and place the title label
title_label = tk.Label(root, text="Extract and analyse online weather data", font=("Helvetica", 16))
title_label.pack(pady=20)

# Create entry fields for start date and end date using DateEntry from tkcalendar
start_date_label = tk.Label(root, text="Start Date:")
start_date_label.pack()

start_date_entry = DateEntry(root, date_pattern='dd/mm/yyyy')
start_date_entry.pack()

end_date_label = tk.Label(root, text="End Date:")
end_date_label.pack()

end_date_entry = DateEntry(root, date_pattern='dd/mm/yyyy')
end_date_entry.pack()

# Create a main frame to hold the LabelFrames
main_frame = tk.Frame(root)
main_frame.pack()

# Create LabelFrames
data_source_frame = tk.LabelFrame(main_frame, text="Data Source")
data_source_frame.pack(fill="both", expand="yes", padx=10, pady=10, side="left")

metrics_frame = tk.LabelFrame(main_frame, text="Metrics")
metrics_frame.pack(fill="both", expand="yes", padx=10, pady=10, side="left")

plot_frame = tk.LabelFrame(main_frame, text="Plot Type")
plot_frame.pack(fill="both", expand="yes", padx=10, pady=10, side="left")

# Create radio buttons for data sources
source_var = tk.StringVar()
source_var.set("KNMI")
knmi_radio = tk.Radiobutton(data_source_frame, text="KNMI", variable=source_var, value="KNMI")
knmi_radio.pack(anchor=tk.W)

meteoblue_radio = tk.Radiobutton(data_source_frame, text="MeteoBlue", variable=source_var, value="MeteoBlue")
meteoblue_radio.pack(anchor=tk.W)

# Create checkbuttons for metrics
precipitation_var = tk.IntVar()
temperature_var = tk.IntVar()
sun_hours_var = tk.IntVar()

precipitation_check = tk.Checkbutton(metrics_frame, text="Precipitation", variable=precipitation_var)
temperature_check = tk.Checkbutton(metrics_frame, text="Temperature", variable=temperature_var)
sun_hours_check = tk.Checkbutton(metrics_frame, text="Sun Hours", variable=sun_hours_var)

precipitation_check.pack(anchor=tk.W)
temperature_check.pack(anchor=tk.W)
sun_hours_check.pack(anchor=tk.W)

# Create checkbuttons for plot types
graph_var = tk.IntVar()
table_var = tk.IntVar()
map_var = tk.IntVar()

graph_check = tk.Checkbutton(plot_frame, text="Graph", variable=graph_var)
table_check = tk.Checkbutton(plot_frame, text="Table", variable=table_var)
map_check = tk.Checkbutton(plot_frame, text="Map", variable=map_var)

graph_check.pack(anchor=tk.W)
table_check.pack(anchor=tk.W)
map_check.pack(anchor=tk.W)

# Create and place the "Go!" button
go_button = tk.Button(root, text="Go!", command=extract_data)
go_button.pack(pady=20)

# Start the main event loop
root.mainloop()
