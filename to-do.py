import that_dawg

# ITS TIME TO REFACTOR

# Gonna split all of this into seperate processes:

### Model loading DONE
### Folder creation DONE
### Basic inference DONE
### Beat counting + trimming DONE
### Beat synced continuation
### Bar counting logic
### Changelog file
### Folders based on generation type

# Flow thoughts:

# Generation tab generates individual loops
### The loops are beat synced and sliced
### Can send audio made here to continuation, assembly

# Continuations tab generates beat synced continuations
### Meant to receive input from generations tab
### Continuations can be fed back into the continuations tab to make more
### Can send audio made here to continuation, assembly

# Assembly tab (probably not in scope for this refactor)
### Collects audio files sent to it from other tabs
### Audio here can be rearanged, looped, and played back
### Basically a mini-daw that's gonna be pretty janky
### Can export the full length final assembly
