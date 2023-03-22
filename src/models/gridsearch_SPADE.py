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

# Load the results
results = pd.read_csv('results.csv')

# Plot the image and pixel accuracies for each class_name and each top_k
for class_name in class_names:
    plt.plot(results[results['class_name'] == class_name]['top_k'], results[results['class_name'] == class_name]['image_accuracy'], label=class_name)
plt.legend()
plt.xlabel('top_k')
plt.ylabel('image_accuracy')
plt.show()
# Save the plot
plt.savefig('grid_image_accuracies.png')

for class_name in class_names:
    plt.plot(results[results['class_name'] == class_name]['top_k'], results[results['class_name'] == class_name]['pixel_accuracy'], label=class_name)
plt.legend()
plt.xlabel('top_k')
plt.ylabel('pixel_accuracy')
plt.show()
# Save the plot
plt.savefig('grid_pixel_accuracy.png')



