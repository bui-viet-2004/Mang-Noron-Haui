# === 6. KIỂM TRA (TEST) MÔ HÌNH ===

import os
os.chdir(os.path.dirname(__file__))

# Thêm hai dòng sau VÀO ĐÂY, trước khi import bất cứ thứ gì khác
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Tắt các thông báo cấp thấp (I) và (W)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Tắt thông báo oneDNN (tùy chọn)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model('best_model.keras')

test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    'test',                     # thư mục chứa ảnh test
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

loss, acc = model.evaluate(test_data)
print(f"🎯 Độ chính xác trên tập test: {acc*100:.2f}%")