# Summary

This project is a visualization to show the differences among the performance of baseball players. It contains four charts, showing the average performance (measured by batting average and home runs) by the players’ weight, height and handedness.

# First Design

The original dataset contains the data of over 1000 players, and each record include three characteristic variables (weight (discrete), height (discrete) and handedness (categorical) and two performance variables (batting average and home runs, both are continuous). In order to show the performance difference of players with different characteristics, I aggregated the data by categorical weight group, height group and handedness, and calculated the average performance for each group.

My first attempt was to use bar charts to show the average performance for players with different characteristics. In each chart, two characteristic variables and one performance variable were displayed.

Then I realized that after the aggregation, one more variable, which is the number of players in each group, was added into the dataset. I used bubble charts to plot the variable. In each chart, two characteristic variables, one performance variable, and the number of players in each group were displayed.


# Feedbacks

Feedback 1: The bubble charts look nicer than the bar charts. However, what the size of the bubbles stand for is not explicit until you move the mouse to it. 

Feedback 2: The bubble charts look nice. However, the bubbles are too scattered and the texts are too small.

Feedback 3: The bubble charts look nice. However, some information is not explicit. What do R, L and B stand for? What's the unit of height and weight? The numbers on the y axis for batting average were not properly displayed.

Feedback 4: Although the bubble charts look "fancy", they are too busy. Sometimes I prefer the variables to be displayed separately so information can be conveyed piece by piece more clearly. 


# Post-feedback designs

Based on the feedbacks and self-reflection, I chose to use bubble charts as it is proper for this dataset and can convey the information I intended to. I also made the following changes and tried different parameters for each changes.
- Revised the number of decimals and the range of ticks on y axis.
- Revised the text to be more explicit. Unit was added, and R, L, B were changed to right handed, left handed and both handed. 
- A note on the size of the bubbles was added.
- the size of the titles were enlarged.
- Two distribution charts of players in different height, weight and handedness group were added.
- Major findings of each chart were added to facilitate the interpretation.

After the changes, the following information was clearly shown in the charts.
- For all the weight/height groups, there are much more right handed players than other types. However, left handed players' average performance are better in most weight/height groups.
- Batting average decreases with increase of the players' height and weight.
- Home runs increases with the players' height and weight for a certain range.

# Resources

http://dimplejs.org/

http://dimplejs.org/examples_viewer.html?id=bubbles_vertical_lollipop

http://dimplejs.org/examples_viewer.html?id=bars_vertical_grouped

https://github.com/PMSI-AlignAlytics/dimple/wiki/dimple.axis#overrideMin

https://github.com/PMSI-AlignAlytics/dimple/wiki/dimple.chart#width

http://stackoverflow.com/questions/27367849/how-to-rotate-and-change-font-size-on-x-axis

http://stackoverflow.com/questions/41618304/how-to-set-the-interval-of-ticks-on-the-x-axis-using-dimple-js
