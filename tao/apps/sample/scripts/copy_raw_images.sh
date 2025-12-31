#!/bin/bash

LABEL_STUDIO_IMAGES_DIR="../../../../label_studio/data/media/upload/1/"
TAO_DATASET_IMAGES_DIR="../data/images/raw/"

mkdir -p "$TAO_DATASET_IMAGES_DIR"

rsync -av --ignore-existing \
  --include="*/" \
  --include="*.jpg" \
  --include="*.jpeg" \
  --include="*.png" \
  --exclude="*" \
  "$LABEL_STUDIO_IMAGES_DIR" \
  "$TAO_DATASET_IMAGES_DIR"

echo "✅ Copia completada (solo imágenes)"
echo "📂 Origen (contenido): $LABEL_STUDIO_IMAGES_DIR"
echo "📂 Destino: $TAO_DATASET_IMAGES_DIR"