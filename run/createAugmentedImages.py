from model.augmentImages import AugmentImages

# --- Run this ONCE only to create augmented data from the available training data ---
augment = AugmentImages()
augment.augmentData()


# --- Run this code to delete the augmented images (comment out the above code before running this) ---
# augment = AugmentImages()
# augment.delAugmentedImages()