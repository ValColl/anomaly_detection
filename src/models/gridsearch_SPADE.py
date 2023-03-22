# Launch  : python3 gridsearch_SPADE.py with different top_k values and show the results

# Import main from SPADE.py
from SPADE import main

# Import pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

class_names = ['bottle', 'wood', 'screw']

# TODO : Optimize this code in order to avoid the repetition of feature extraction. Did the tests with it anyway.
for top_k in [2, 5, 10]:
    # Launch main with top_k and class_names
    main(top_k=top_k, class_names=class_names)

## Load the results
# Create a double dictionary to store the results with keys class_name and top_k
image_rocauc = {}
pixel_rocauc = {}
for class_name in class_names:
    image_rocauc[class_name] = {}
    pixel_rocauc[class_name] = {}
avg_image_rocauc = {}
avg_pixel_rocauc = {}
# Read each line of the file and store it in a list
with open('../../models/SPADE/results.csv', 'r') as f:
    lines = f.readlines()
# Iterate over the lines
for line in lines[1:]:
    # Split the line by spaces
    line = line.strip().split(',')
    print(line)
    # Get the class name, top_k, and the AUC value
    top_k = int(line[0])
    for i, class_name in enumerate(class_names):
        image_rocauc[class_name][top_k] = float(line[2*i+1])
        pixel_rocauc[class_name][top_k] = float(line[2*i+2])
    avg_image_rocauc[top_k] = float(line[-2])
    avg_pixel_rocauc[top_k] = float(line[-1])


print(image_rocauc)

# Plot the image_rocauc and average_rocauc for each class
for class_name in class_names:
    plt.plot(list(image_rocauc[class_name].keys()), list(image_rocauc[class_name].values()), label=class_name)
plt.plot(list(avg_image_rocauc.keys()), list(avg_image_rocauc.values()), label='Average')
plt.xlabel('top_k')
plt.ylabel('Image ROC AUC')
plt.title('Image ROC AUC for each class')
plt.legend()

# Save the plot
plt.savefig('../../models/SPADE/image_rocauc.png')
