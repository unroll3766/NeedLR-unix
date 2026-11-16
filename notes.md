1\) Warning on python main.py launch  
objc\[22187\]: Class QT\_ROOT\_LEVEL\_POOL\_\_THESE\_OBJECTS\_WILL\_BE\_RELEASED\_WHEN\_QAPP\_GOES\_OUT\_OF\_SCOPE is implemented in both /Users/theo/anaconda3/envs/NeedLR/lib/libQt5Core.5.15.8.dylib (0x104e952e0) and /Users/theo/anaconda3/envs/NeedLR/lib/libQt6Core.6.7.2.dylib (0x1671a5558). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.  
objc\[22187\]: Class KeyValueObserver is implemented in both /Users/theo/anaconda3/envs/NeedLR/lib/libQt5Core.5.15.8.dylib (0x104e95308) and /Users/theo/anaconda3/envs/NeedLR/lib/libQt6Core.6.7.2.dylib (0x1671a5580). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.  
objc\[22187\]: Class RunLoopModeTracker is implemented in both /Users/theo/anaconda3/envs/NeedLR/lib/libQt5Core.5.15.8.dylib (0x104e95358) and /Users/theo/anaconda3/envs/NeedLR/lib/libQt6Core.6.7.2.dylib (0x1671a55d0). This may cause spurious casting failures and mysterious crashes. One of the duplicates must be removed or renamed.  
Current subsampling found from settings:  random  
index on subsampling:  0  
Current spg found from settings:  knn  
index on spg:  0

2\) GUI displays "loading..." when an invalid directory is selected

3\) Program crashes when a non-int is used for label

4\) Window is taller than my screen (due to task bar)

5\) Need a .laz reader

6\) Need a way to apply the labels to the original point cloud with the JSON

Thoughts

1\) It's hard to distinguish between some of the DWD due to the downsampling. Changing background point size might be helpful.

2\) A dropdown with prior classes (or a link to a pre-chosen schema) might be helpful