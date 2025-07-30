import cv2
import numpy as np

# 1) Carga imágenes
# Sustituye estas rutas por las tuyas (pueden ser .tif, .png, .jpg…)
orig = cv2.imread('data/IN/proan_pivote.tif', cv2.IMREAD_COLOR)
mask = cv2.imread('weed_pred_proan_pivote.tif', cv2.IMREAD_GRAYSCALE)

# 2) Normaliza la máscara a 0-255
mask_norm = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 3) Aplica un colormap (JET, HOT, etc.)
heatmap = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)

# 4) Superponer con transparencia
alpha = 0.5  # 0.0 = sólo original, 1.0 = sólo heatmap
overlay = cv2.addWeighted(orig, 1-alpha, heatmap, alpha, 0)

# 5) Muestra o guarda
# cv2.imshow('Heatmap Overlay', overlay)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Para guardar:
cv2.imwrite('overlay_heatmap.png', overlay)
