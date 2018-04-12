# roomDetector
using sentdex tutorial + tf + hashtables to predict rooms.
when run, after 20 seconds+ of loading weights, it will begin to show features for your current computer screen.
Should be stored in 'models/research/object_detection' so it has access to frozen mobilenet model and visualisation_utils.py
sentdex.py is main file, it displays images, loads into graph and calls methods of memoryCache.
memoryCache is responsible for reinforcement.
