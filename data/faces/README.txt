Put your face images here.

Folder structure:
  data/faces/
    user1/
      0001.jpg
      0002.jpg
      ...
    user2/
      0001.jpg
      0002.jpg
      ...

Then run:
  python scripts/build_dataset.py

Or capture directly from webcam:
  python scripts/capture_faces.py --identity user1 --count 60
  python scripts/capture_faces.py --identity user2 --count 60
