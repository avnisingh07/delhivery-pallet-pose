import cv2

from section2_pose.inference.section1_adapter import (
    Section1GeometryAdapter,
)


IMAGE_PATH = "data/raw/dataset_2/train/images/d7c3089e-test_image_44_jpg.rf.25a6ff21418842e392582bb2c80a2b8c.jpg"


def main():

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    adapter = Section1GeometryAdapter(
        conf_threshold=0.25
    )

    outputs = adapter.predict(image)

    print(f"Detected geometry instances: {len(outputs)}")

    for item in outputs:

        print(
            f"instance={item['instance_id']} "
            f"confidence={item['confidence']:.3f} "
            f"G0={item['G0']} "
            f"G1={item['G1']}"
        )


if __name__ == "__main__":
    main()