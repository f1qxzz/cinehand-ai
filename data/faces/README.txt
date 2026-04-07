Put your face images here.

Folder structure:
  data/faces/
    afna/
      0001.jpg
      0002.jpg
      ...
    f1qxzz/
      0001.jpg
      0002.jpg
      ...

Then run:
  python scripts/build_dataset.py

Or capture directly from webcam:
  python scripts/capture_faces.py --identity afna --count 60
  python scripts/capture_faces.py --identity f1qxzz --count 60
