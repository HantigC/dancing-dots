import numpy as np


def transform_keypoints_to_original(
    kpts_crop: np.ndarray,
    original_size: tuple[int, int],
    size_param: int = 512,
    square_ok: bool = False,
) -> np.ndarray:
    """
    Transforms keypoint coordinates from a DUST3R-processed (resized and cropped)
    image back to the original image's coordinate system.

    Args:
        kpts_crop: A NumPy array of shape (N, 2) where N is the number of keypoints,
                   and each row is (x, y) coordinate on the processed image.
        original_size: A tuple (original_width, original_height) of the original image.
        resized_crop_size: A tuple (processed_width, processed_height) of the
                           image after resizing and cropping (i.e., the dimensions
                           of the input image to DUST3R). This is W2, H2 from the
                           load_images function.
        size_param: The 'size' parameter (e.g., 224, 512) used in the
                    original load_images function.
        square_ok: The 'square_ok' parameter used in the original load_images function.

    Returns:
        A NumPy array of shape (N, 2) with the transformed keypoint coordinates
        on the original image.
    """
    # print(f"original_size: {original_size}")
    original_height, original_width = original_size
    original_height = float(original_height)
    original_width = float(original_width)

    # --- 1. Determine the dimensions after resizing but *before* cropping (W_res, H_res) ---
    # This logic mirrors the _resize_pil_image call in load_images
    if size_param == 224:
        # Target long side is used for resizing.
        target_long_side = round(
            size_param
            * max(original_width / original_height, original_height / original_width)
        )
        if original_width >= original_height:
            W_res = target_long_side
            H_res = round(original_height * (target_long_side / original_width))
        else:
            H_res = target_long_side
            W_res = round(original_width * (target_long_side / original_height))
    else:
        # Long side is resized to size_param.
        if original_width >= original_height:
            W_res = size_param
            H_res = round(original_height * (size_param / original_width))
        else:
            H_res = size_param
            W_res = round(original_width * (size_param / original_height))

    # print(f"H_res, W_res: {H_res}_{W_res}")

    # --- 2. Calculate the cropping offsets used during processing ---
    cx, cy = W_res // 2, H_res // 2

    if size_param == 224:
        half = min(cx, cy)
        crop_left = cx - half
        crop_top = cy - half
    else:
        halfw = ((2 * cx) // 16) * 8
        halfh = ((2 * cy) // 16) * 8
        if not square_ok and W_res == H_res:
            halfh = round(3 * halfw / 4)

        crop_left = cx - halfw
        crop_top = cy - halfh

    # --- 4. Reverse the Resizing ---
    # Determine the actual scaling factor applied during the initial resize
    if original_width >= original_height:
        scale_factor = size_param / original_width
    else:
        scale_factor = size_param / original_height
    # --- 3. Reverse the Cropping ---
    # Add the crop offsets to the keypoints from the cropped image
    # print(crop_left, crop_top)
    kpts_resized = kpts_crop.astype(float)  # Ensure float for accurate division
    kpts_resized[:, 0] = kpts_resized[:, 0] + crop_left
    kpts_resized[:, 1] = kpts_resized[:, 1] + crop_top

    # Divide by the scale factor to get original coordinates
    kpts_original = kpts_resized / scale_factor

    return kpts_original
