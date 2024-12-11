import random

import pandas as pd


def split_train_test(
    pkl_file,
    test_ratio=0.1,
):
    """
    Splits the dataset into training and testing sets such that no patient overlaps
    between the training and testing sets.

    Parameters:
    - pkl_file: Path to the input pickle file containing the dataset.
    - train_img_out: Path to save the training images pickle.
    - test_img_out: Path to save the testing images pickle.
    - train_label_out: Path to save the training labels pickle.
    - test_label_out: Path to save the testing labels pickle.
    - test_ratio: Fraction of the dataset to allocate for testing.

    Returns:
    - None
    """
    path = "./data/processed/mri"
    train_img_out=f"{path}/img_train.pkl"
    test_img_out=f"{path}/img_test.pkl"
    train_label_out=f"{path}/img_y_train.pkl"
    test_label_out=f"{path}/img_y_test.pkl"

    # Load the dataset
    m2 = pd.read_pickle(pkl_file)

    # Clean patient IDs (optional, based on your dataset format)
    m2["subject"] = m2["subject"].str.replace("s", "S").str.replace("\n", "")

    # Get unique patient IDs
    subjects = list(set(m2["subject"].values))
    print(f"Total unique patients: {len(subjects)}")

    # Calculate the number of patients for the test set
    num_test_subjects = int(test_ratio * len(subjects))
    print(f"Number of patients in the test set: {num_test_subjects}")

    # Randomly pick patient IDs for the test set
    picked_ids = random.sample(subjects, num_test_subjects)

    # Create the test set
    test = pd.DataFrame(columns=m2.columns)
    for pid in picked_ids:
        # Pick one MRI for each patient to avoid repetition
        s = m2[m2["subject"] == pid].sample()
        test = pd.concat([test, s], ignore_index=True)

    # Get the remaining indexes for the training set
    test_indexes = set(test.index)
    train = m2[~m2.index.isin(test_indexes)]

    # Save the training and testing data
    train[["img_array"]].to_pickle(train_img_out)
    test[["img_array"]].to_pickle(test_img_out)
    train[["label"]].to_pickle(train_label_out)
    test[["label"]].to_pickle(test_label_out)

    print("Train and test sets created and saved.")
    print(f"Training set size: {len(train)}")
    print(f"Testing set size: {len(test)}")


def main():
    path = "./data/processed/mri"
    split_train_test(
        pkl_file=f"{path}/mri_meta.pkl",
        test_ratio=0.1,
    )

if __name__ == "__main__":
    main()
