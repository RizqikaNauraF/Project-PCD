# import streamlit as st
# import cv2
# import numpy as np
# import joblib

# def manual_histogram_equalization(img):
#     hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
#     cdf = hist.cumsum()
#     cdf_normalized = cdf / (cdf[-1] + 1e-6)
#     equalized_img = np.interp(img.flatten(), bins[:-1], cdf_normalized * 255)
#     return equalized_img.reshape(img.shape).astype(np.uint8)

# def manual_gaussian_blur(img, kernel_size=5, sigma=1.0):
#     ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
#     xx, yy = np.meshgrid(ax, ax)
#     kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
#     kernel = kernel / np.sum(kernel)

#     pad = kernel_size // 2
#     padded_img = np.pad(img, ((pad, pad), (pad, pad)), mode='constant')
#     blurred = np.zeros_like(img)

#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             region = padded_img[i:i+kernel_size, j:j+kernel_size]
#             blurred[i, j] = np.sum(region * kernel)

#     return blurred.astype(np.uint8)

# def manual_otsu_threshold(img):
#     hist, bins = np.histogram(img.flatten(), bins=256, range=[0,256])
#     total = img.size
#     current_max, threshold = 0, 0
#     sum_total, sum_foreground, weight_background = 0, 0, 0
#     weight_foreground = 0

#     for i in range(256):
#         sum_total += i * hist[i]
#     for i in range(256):
#         weight_background += hist[i]
#         if weight_background == 0:
#             continue
#         weight_foreground = total - weight_background
#         if weight_foreground == 0:
#             break
#         sum_foreground += i * hist[i]
#         mean_background = sum_foreground / weight_background
#         mean_foreground = (sum_total - sum_foreground) / weight_foreground
#         var_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
#         if var_between > current_max:
#             current_max = var_between
#             threshold = i

#     binary_img = np.where(img > threshold, 255, 0).astype(np.uint8)
#     return binary_img, threshold

# def erosion(img, kernel):
#     img_h, img_w = img.shape
#     k_h, k_w = kernel.shape
#     pad_h, pad_w = k_h // 2, k_w // 2
#     padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)
#     eroded = np.zeros_like(img)
#     for i in range(img_h):
#         for j in range(img_w):
#             region = padded_img[i:i+k_h, j:j+k_w]
#             eroded[i, j] = np.min(region[kernel==1])
#     return eroded

# def dilation(img, kernel):
#     img_h, img_w = img.shape
#     k_h, k_w = kernel.shape
#     pad_h, pad_w = k_h // 2, k_w // 2
#     padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
#     dilated = np.zeros_like(img)
#     for i in range(img_h):
#         for j in range(img_w):
#             region = padded_img[i:i+k_h, j:j+k_w]
#             dilated[i, j] = np.max(region[kernel==1])
#     return dilated

# def morphological_operations_manual(img, kernel_size=(5, 5)):
#     kernel = np.ones(kernel_size, dtype=np.uint8)
#     opened = dilation(erosion(img, kernel), kernel)
#     closed = erosion(dilation(img, kernel), kernel)
#     return opened, closed

# def manual_lbp_with_mask(image, mask):
#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     lbp_image = np.zeros_like(gray)
#     for i in range(1, gray.shape[0] - 1):
#         for j in range(1, gray.shape[1] - 1):
#             if mask[i, j] == 0:
#                 continue
#             center = gray[i, j]
#             binary = ''
#             binary += '1' if gray[i-1, j-1] >= center else '0'
#             binary += '1' if gray[i-1, j] >= center else '0'
#             binary += '1' if gray[i-1, j+1] >= center else '0'
#             binary += '1' if gray[i, j+1] >= center else '0'
#             binary += '1' if gray[i+1, j+1] >= center else '0'
#             binary += '1' if gray[i+1, j] >= center else '0'
#             binary += '1' if gray[i+1, j-1] >= center else '0'
#             binary += '1' if gray[i, j-1] >= center else '0'
#             lbp_val = int(binary, 2)
#             lbp_image[i, j] = lbp_val

#     masked_lbp = lbp_image[mask == 255]
#     hist, _ = np.histogram(masked_lbp, bins=256, range=(0, 256), density=True)
#     return hist, lbp_image

# def extract_feature_from_uploaded_image(img_gray):
#     img_resized = cv2.resize(img_gray, (256, 256))
#     eq_img = manual_histogram_equalization(img_resized)
#     blurred = manual_gaussian_blur(eq_img)
#     otsu_img, _ = manual_otsu_threshold(blurred)
#     _, closed_mask = morphological_operations_manual(otsu_img)
#     hist, _ = manual_lbp_with_mask(img_resized, closed_mask)
#     return hist, closed_mask, img_resized

# # Prediksi Manual
# def dot(w, x):
#     return sum(w_i * x_i for w_i, x_i in zip(w, x))

# def predict_linear_loaded(X):
#     return [1 if dot(w_linear_loaded, x) + b_linear_loaded >= 0 else -1 for x in X]

# # Load Model
# model_path = "svm_linear_model.pkl"
# try:
#     model_linear_loaded = joblib.load(model_path)
#     w_linear_loaded = model_linear_loaded['weights']
#     b_linear_loaded = model_linear_loaded['bias']
# except FileNotFoundError:
#     st.error("Model tidak ditemukan. Pastikan file `svm_linear_model.pkl` ada di direktori yang sama.")
#     st.stop()

# #Streamlit UI
# st.title("Deteksi Tuberkulosis (TBC) pada Citra X-ray Paru-paru")

# uploaded_file = st.file_uploader("Unggah gambar X-ray paru-paru", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     original_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

#     feature_vector, mask, img_resized = extract_feature_from_uploaded_image(original_img)
#     pred = predict_linear_loaded([feature_vector])[0]
#     label = "Normal" if pred == -1 else "TBC"

#     original_rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
#     overlay = original_rgb_img.copy()
#     overlay[mask > 0] = [0, 255, 0] if label == "Normal" else [255, 0, 0]
#     combined_img = cv2.addWeighted(overlay, 0.3, original_rgb_img, 0.7, 0)

#     st.subheader("Hasil Prediksi")
#     st.markdown(f"**Prediksi Model SVM Linear:** `{label}`")
#     st.image(combined_img, caption=f"Hasil Segmentasi dan Prediksi: {label}", channels="RGB", use_container_width=True)


import streamlit as st
import cv2
import numpy as np
import joblib


# Navigasi Halaman
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman", ["📘 Tutorial Penggunaan", "🔬 Deteksi TBC"])


# Halaman 1: Tutorial
if page == "📘 Tutorial Penggunaan":
    st.title("📘 Tutorial Penggunaan Aplikasi Deteksi TBC")
    st.markdown("""
    Selamat datang di Sistem **Deteksi Tuberkulosis (TBC)** berbasis citra X-ray!

    ### Langkah-langkah Penggunaan:
    1. Buka tab **"Deteksi TBC"** di sidebar.
    2. Unggah gambar X-ray paru-paru dengan format `.jpg`, `.jpeg`, atau `.png`.
    3. Sistem akan melakukan preprocessing, segmentasi, dan ekstraksi fitur.
    4. Model SVM akan memprediksi apakah gambar tersebut **Normal** atau **TBC**.
    5. Hasil segmentasi dan label prediksi akan ditampilkan secara visual.

    > **Catatan:** Gunakan gambar X-ray resolusi cukup (256x256 atau lebih) agar hasil maksimal.
    """)
    st.markdown("""
                <hr style="border: 0.5px solid #ccc;" />
                <center><small>&copy; 2025 Kelompok 5 - Project Pengolahan Citra Digital | Universitas Negeri Surabaya</small></center>
                """, unsafe_allow_html=True)



# Halaman 2: Deteksi TBC
elif page == "🔬 Deteksi TBC":

    st.title("Deteksi Tuberkulosis (TBC) pada Citra X-ray Paru-paru")

    # Fungsi preprocessing & fitur (dari sebelumnya)
    def manual_histogram_equalization(img):
        hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / (cdf[-1] + 1e-6)
        equalized_img = np.interp(img.flatten(), bins[:-1], cdf_normalized * 255)
        return equalized_img.reshape(img.shape).astype(np.uint8)

    def manual_gaussian_blur(img, kernel_size=5, sigma=1.0):
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / np.sum(kernel)

        pad = kernel_size // 2
        padded_img = np.pad(img, ((pad, pad), (pad, pad)), mode='constant')
        blurred = np.zeros_like(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded_img[i:i+kernel_size, j:j+kernel_size]
                blurred[i, j] = np.sum(region * kernel)

        return blurred.astype(np.uint8)

    def manual_otsu_threshold(img):
        hist, bins = np.histogram(img.flatten(), bins=256, range=[0,256])
        total = img.size
        current_max, threshold = 0, 0
        sum_total, sum_foreground, weight_background = 0, 0, 0
        weight_foreground = 0

        for i in range(256):
            sum_total += i * hist[i]
        for i in range(256):
            weight_background += hist[i]
            if weight_background == 0:
                continue
            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break
            sum_foreground += i * hist[i]
            mean_background = sum_foreground / weight_background
            mean_foreground = (sum_total - sum_foreground) / weight_foreground
            var_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            if var_between > current_max:
                current_max = var_between
                threshold = i

        binary_img = np.where(img > threshold, 255, 0).astype(np.uint8)
        return binary_img, threshold

    def erosion(img, kernel):
        img_h, img_w = img.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)
        eroded = np.zeros_like(img)
        for i in range(img_h):
            for j in range(img_w):
                region = padded_img[i:i+k_h, j:j+k_w]
                eroded[i, j] = np.min(region[kernel==1])
        return eroded

    def dilation(img, kernel):
        img_h, img_w = img.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        dilated = np.zeros_like(img)
        for i in range(img_h):
            for j in range(img_w):
                region = padded_img[i:i+k_h, j:j+k_w]
                dilated[i, j] = np.max(region[kernel==1])
        return dilated

    def morphological_operations_manual(img, kernel_size=(5, 5)):
        kernel = np.ones(kernel_size, dtype=np.uint8)
        opened = dilation(erosion(img, kernel), kernel)
        closed = erosion(dilation(img, kernel), kernel)
        return opened, closed

    def manual_lbp_with_mask(image, mask):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp_image = np.zeros_like(gray)
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                if mask[i, j] == 0:
                    continue
                center = gray[i, j]
                binary = ''
                binary += '1' if gray[i-1, j-1] >= center else '0'
                binary += '1' if gray[i-1, j] >= center else '0'
                binary += '1' if gray[i-1, j+1] >= center else '0'
                binary += '1' if gray[i, j+1] >= center else '0'
                binary += '1' if gray[i+1, j+1] >= center else '0'
                binary += '1' if gray[i+1, j] >= center else '0'
                binary += '1' if gray[i+1, j-1] >= center else '0'
                binary += '1' if gray[i, j-1] >= center else '0'
                lbp_val = int(binary, 2)
                lbp_image[i, j] = lbp_val

        masked_lbp = lbp_image[mask == 255]
        hist, _ = np.histogram(masked_lbp, bins=256, range=(0, 256), density=True)
        return hist, lbp_image

    def extract_feature_from_uploaded_image(img_gray):
        img_resized = cv2.resize(img_gray, (256, 256))
        eq_img = manual_histogram_equalization(img_resized)
        blurred = manual_gaussian_blur(eq_img)
        otsu_img, _ = manual_otsu_threshold(blurred)
        _, closed_mask = morphological_operations_manual(otsu_img)
        hist, _ = manual_lbp_with_mask(img_resized, closed_mask)
        return hist, closed_mask, img_resized

    def dot(w, x):
        return sum(w_i * x_i for w_i, x_i in zip(w, x))

    def predict_linear_loaded(X):
        return [1 if dot(w_linear_loaded, x) + b_linear_loaded >= 0 else -1 for x in X]

    # Load model
    model_path = "svm_linear_model.pkl"
    try:
        model_linear_loaded = joblib.load(model_path)
        w_linear_loaded = model_linear_loaded['weights']
        b_linear_loaded = model_linear_loaded['bias']
    except FileNotFoundError:
        st.error("Model tidak ditemukan. Pastikan `svm_linear_model.pkl` ada di direktori yang sama.")
        st.stop()

    # Upload gambar
    uploaded_file = st.file_uploader("Unggah gambar X-ray paru-paru", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        feature_vector, mask, img_resized = extract_feature_from_uploaded_image(original_img)
        pred = predict_linear_loaded([feature_vector])[0]
        label = "Normal" if pred == -1 else "TBC"

        original_rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        overlay = original_rgb_img.copy()
        overlay[mask > 0] = [0, 255, 0] if label == "Normal" else [255, 0, 0]
        combined_img = cv2.addWeighted(overlay, 0.3, original_rgb_img, 0.7, 0)

        st.subheader("Hasil Prediksi")
        st.markdown(f"**Prediksi Model SVM Linear:** `{label}`")
        st.image(combined_img, caption=f"Hasil Segmentasi dan Prediksi: {label}", channels="RGB", use_container_width=True)
    st.markdown("""
                <hr style="border: 0.5px solid #ccc;" />
                <center><small>&copy; 2025 Kelompok 5 - Project Pengolahan Citra Digital | Universitas Negeri Surabaya</small></center>
                """, unsafe_allow_html=True)

