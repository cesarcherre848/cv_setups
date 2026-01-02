import os
import glob
import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_CLASS_COLORS = {
    0: None,              # fondo (no se pinta)
    1: (255, 0, 0),       # azul
    2: (0, 255, 0),       # verde
    3: (0, 0, 255),       # rojo
    4: (255, 255, 0),     # cyan
    5: (255, 0, 255),     # magenta
    6: (0, 255, 255),     # amarillo
    7: (128, 0, 0),       # azul oscuro
    8: (0, 128, 0),       # verde oscuro
    9: (0, 0, 128),       # rojo oscuro
}


def get_class_color_map(num_classes, class_colors=None):
    """
    Retorna un array [num_classes, 3] en BGR.
    La clase 0 queda negra (fondo).
    """

    if class_colors is None:
        class_colors = DEFAULT_CLASS_COLORS

    colors = np.zeros((num_classes, 3), dtype=np.uint8)

    for cls_id in range(num_classes):
        if cls_id in class_colors and class_colors[cls_id] is not None:
            colors[cls_id] = class_colors[cls_id]
        else:
            colors[cls_id] = (0, 0, 0)  # fallback

    return colors


def segformer_infer_overlay(
    model_path,
    image_inputs_paths,
    image_output_dir,
    input_size=(512, 512),
    alpha=0.4,
):
    """
    Ejecuta inferencia SegFormer ONNX y genera overlay multiclass.

    Args:
        model_path (str): path al modelo ONNX
        image_inputs_paths (list[str]): lista de paths de imágenes
        image_output_dir (str): carpeta de salida
        input_size (tuple): tamaño de entrada del modelo (W, H)
        alpha (float): transparencia de la máscara
    """

    os.makedirs(image_output_dir, exist_ok=True)

    # ---------------------------
    # 1. Cargar modelo (una sola vez)
    # ---------------------------
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Normalización SegFormer
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)



    for img_path in image_inputs_paths:

        # ---------------------------
        # 2. Leer imagen ORIGINAL
        # ---------------------------
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[WARN] No se pudo leer {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # ---------------------------
        # 3. Preprocesado
        # ---------------------------
        img = cv2.resize(img_rgb, input_size)
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0)

        # ---------------------------
        # 4. Inferencia
        # ---------------------------
        output = session.run([output_name], {input_name: img})[0]
        mask = np.argmax(output, axis=1)[0].astype(np.uint8)

        # ---------------------------
        # 5. Reescalar máscara
        # ---------------------------
        mask_resized = cv2.resize(
            mask,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )

        # ---------------------------
        # 6. Colorizar máscara
        # ---------------------------
        num_classes = output.shape[1]

        colors = get_class_color_map(num_classes)
        mask_color = colors[mask_resized]

        overlay = img_bgr.copy()
        mask_bin = mask_resized != 0

        overlay[mask_bin] = (
            overlay[mask_bin].astype(np.float32) * (1 - alpha)
            + mask_color[mask_bin].astype(np.float32) * alpha
        ).astype(np.uint8)

        # ---------------------------
        # 7. Guardar resultado
        # ---------------------------
        out_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(image_output_dir, f"{out_name}_overlay.png")

        cv2.imwrite(out_path, overlay)
        print(f"[OK] {out_path}")


image_dir = "../data/dataset/images/val"

images = sorted(
    glob.glob(os.path.join(image_dir, "*.jpg"))
)



segformer_infer_overlay(
    model_path="../results/export/segformer.onnx",
    image_inputs_paths=images,
    image_output_dir="../data/dataset/fancy_results",
    alpha=0.4
)