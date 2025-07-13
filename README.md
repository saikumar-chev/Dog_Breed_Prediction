# Dog Breed Identification App

A deep learning project to identify dog breeds from images.

## Setup

1. Install requirements:
    ```
    pip install -r requirements.txt
    ```
2. Prepare your dataset in `data/train` (one folder per breed).

3. Train the model:
    ```
    python train.py
    ```
4. Run the app:
    ```
    streamlit run app.py
    ```
## Notes

- Update `class_indices` in `app.py` to match your dataset's classes.
- The model uses MobileNetV2 for transfer learning.