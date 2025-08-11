import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import imagehash
from finalsignature import crop_signature
from finalcheque import tightest_signature_crop
from skimage.metrics import structural_similarity as ssim

# --- Page Config ---
st.set_page_config(page_title="🖊️ Signature Verifier", layout="wide")

# --- Custom CSS for Aesthetics ---
st.markdown("""
    <style>
    .stImage img {
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .center-text {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def extract_signature(image_file, mode):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_input:
        image_file.save(temp_input.name)
        input_path = temp_input.name
    if mode == "White Background Signature":
        return crop_signature(input_path)
    else:
        return tightest_signature_crop(input_path)

def compare_images(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_resized = cv2.resize(img1_gray, (300, 150))
    img2_resized = cv2.resize(img2_gray, (300, 150))
    score, _ = ssim(img1_resized, img2_resized, full=True)
    return score

def hash_difference(img1, img2):
    img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    hash1 = imagehash.average_hash(img1_pil)
    hash2 = imagehash.average_hash(img2_pil)
    return abs(hash1 - hash2)

# --- Header ---
st.markdown(
    """
    <div class="center-text">
        <h1 style="color:#2c3e50;">🖼️ Signature Verifier</h1>
        <p style="color:#7f8c8d; font-size:18px;">
            Upload images to extract or compare signatures using AI-powered tools.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Signature Tool")
    mode = st.radio("Choose Mode", ["Extract Signature", "Compare Signatures"])

# --- Extract Signature Mode ---
if mode == "Extract Signature":
    st.subheader("✂️ Extract Signature from an Image")

    col1, col2 = st.columns([1, 2])
    with col1:
        img_type = st.radio("Image Type", ["White Background Signature", "Cheque Image"])
        uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        with st.spinner("🔍 Extracting signature..."):
            result = extract_signature(image, img_type)

        if result is not None:
            st.success("✅ Signature extracted successfully!")

            # Display images side-by-side
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(image, caption="📄 Original Image", width=250)
            with img_col2:
                st.image(result, caption="✂️ Extracted Signature", width=200)

            # Download button
            result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_output:
                result_pil.save(temp_output.name)
                st.download_button(
                    label="⬇️ Download Signature",
                    data=open(temp_output.name, "rb").read(),
                    file_name="processed_signature.png",
                    mime="image/png"
                )
        else:
            st.error("❌ Signature not found in the image.")

# --- Compare Signatures Mode ---
else:
    st.subheader("🔍 Compare Two Signatures")
    st.markdown("Upload a **reference signature** and a **test signature** below:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🧾 Reference Image")
        ref_type = st.radio("Type", ["White Background Signature", "Cheque Image"], key="ref_type")
        ref_file = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"], key="ref_upload")

    with col2:
        st.markdown("##### 📝 Test Image")
        test_type = st.radio("Type", ["White Background Signature", "Cheque Image"], key="test_type")
        test_file = st.file_uploader("Upload Test Image", type=["png", "jpg", "jpeg"], key="test_upload")

    if ref_file and test_file:
        ref_img = Image.open(ref_file)
        test_img = Image.open(test_file)

        with st.spinner("✂️ Extracting signatures..."):
            ref_sig = extract_signature(ref_img, ref_type)
            test_sig = extract_signature(test_img, test_type)

        if ref_sig is None or test_sig is None:
            st.error("❌ Could not extract signatures from one or both images.")
        else:
            st.success("✅ Both signatures extracted successfully!")

            # Show side-by-side comparison
            st.markdown("#### 🔍 Side-by-Side Comparison")
            col1, col2 = st.columns(2)
            col1.image(ref_sig, caption="Reference Signature", width=200)
            col2.image(test_sig, caption="Test Signature", width=200)

            with st.spinner("🔍 Comparing..."):
                similarity_score = compare_images(ref_sig, test_sig)
                hash_diff = hash_difference(ref_sig, test_sig)

            # Results
            match_threshold = 0.75
            hash_threshold = 10

            st.markdown("---")
            st.markdown("### 🧪 Comparison Results")
            st.metric(label="SSIM Similarity", value=f"{similarity_score:.2f}")
            st.metric(label="Hash Difference", value=f"{hash_diff}")

            if similarity_score >= match_threshold and hash_diff <= hash_threshold:
                st.success(f"✅ Match Found with {similarity_score*100:.2f}% similarity")
            else:
                st.error("❌ Signatures do not match confidently.")
