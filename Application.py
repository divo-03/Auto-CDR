import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

st.set_page_config(page_title="Auto CDR", layout="wide")
st.title("Auto CDR")
st.subheader("Upload a fundus image. The AI will automatically segment the optic cup and disc and compute the cup-to-disc ratio (CDR)")

@st.cache_resource 
def load_model():
    import segmentation_models_pytorch as smp
    import torch
    
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b4', 
        encoder_weights=None, 
        in_channels=3, 
        classes=3, 
        activation=None
    )
    
    model.load_state_dict(torch.load(None, map_location=torch.device('cpu'), weights_only=False))
    model.eval()
    return model

model = load_model()

def Preprocessing(image_np, clip_limit=4.0):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def clean_mask_and_get_height(mask):
    mask_uint8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_smoothed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_smoothed = cv2.morphologyEx(mask_smoothed, cv2.MORPH_OPEN, kernel)
    blurred = cv2.GaussianBlur(mask_smoothed, (15, 15), 0)
    _, mask_smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0, None 
        
    largest_contour = max(contours, key=cv2.contourArea)
    _, _, _, h = cv2.boundingRect(largest_contour)
    return h, largest_contour

def overlay_masks(image_np, disc_contour, cup_contour):
    overlay_fill = image_np.copy()
    img_h, img_w = image_np.shape[:2]
    
    if disc_contour is not None and len(disc_contour) >= 5:
        disc_ellipse = cv2.fitEllipse(disc_contour)
        cv2.ellipse(overlay_fill, disc_ellipse, (255, 0, 0), -1) 
    elif disc_contour is not None:
        cv2.drawContours(overlay_fill, [disc_contour], -1, (255, 0, 0), -1)
        
    if cup_contour is not None and len(cup_contour) >= 5:
        cup_ellipse = cv2.fitEllipse(cup_contour)
        cv2.ellipse(overlay_fill, cup_ellipse, (0, 0, 255), -1)
    elif cup_contour is not None:
        cv2.drawContours(overlay_fill, [cup_contour], -1, (0, 0, 255), -1)
        
    alpha = 0.5 
    blended = cv2.addWeighted(overlay_fill, alpha, image_np, 1 - alpha, 0)
    
    if disc_contour is not None:
        disc_x, _, disc_w, _ = cv2.boundingRect(disc_contour)
        od_caliper_x = max(5, disc_x - int(disc_w * 0.15))
        oc_caliper_x = min(img_w - 5, disc_x + disc_w + int(disc_w * 0.15))
    else:
        od_caliper_x = 10
        oc_caliper_x = img_w - 30

    if disc_contour is not None and len(disc_contour) >= 5:
        disc_ellipse = cv2.fitEllipse(disc_contour)
        cv2.ellipse(blended, disc_ellipse, (255, 0, 0), 2) 
        
        x, y, w, h = cv2.boundingRect(disc_contour)
        touch_x = x + w // 2
        
        cv2.line(blended, (od_caliper_x, y), (od_caliper_x, y + h), (255, 0, 0), 2)
        cv2.line(blended, (od_caliper_x, y), (touch_x, y), (255, 0, 0), 2)
        cv2.line(blended, (od_caliper_x, y + h), (touch_x, y + h), (255, 0, 0), 2)
        
        cv2.putText(blended, "OD", (max(0, od_caliper_x - 35), y + h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    if cup_contour is not None and len(cup_contour) >= 5:
        cup_ellipse = cv2.fitEllipse(cup_contour)
        cv2.ellipse(blended, cup_ellipse, (0, 0, 255), 2)
        
        x, y, w, h = cv2.boundingRect(cup_contour)
        touch_x = x + w // 2
        
        cv2.line(blended, (oc_caliper_x, y), (oc_caliper_x, y + h), (0, 0, 255), 2)
        cv2.line(blended, (oc_caliper_x, y), (touch_x, y), (0, 0, 255), 2)
        cv2.line(blended, (oc_caliper_x, y + h), (touch_x, y + h), (0, 0, 255), 2)
        
        cv2.putText(blended, "OC", (min(img_w - 35, oc_caliper_x + 8), y + h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    return blended

def generate_raw_mask_visual(h, w, disc_contour, cup_contour):
    mask_rgb = np.full((h, w, 3), 255, dtype=np.uint8) 
    
    if disc_contour is not None and len(disc_contour) >= 5:
        disc_ellipse = cv2.fitEllipse(disc_contour)
        cv2.ellipse(mask_rgb, disc_ellipse, (128, 128, 128), -1)
        
    if cup_contour is not None and len(cup_contour) >= 5:
        cup_ellipse = cv2.fitEllipse(cup_contour)
        cv2.ellipse(mask_rgb, cup_ellipse, (0, 0, 0), -1)
        
    return mask_rgb


uploaded_file = st.file_uploader("Upload Fundus", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    enhanced_np = Preprocessing(image_np, clip_limit=4.0)
    
    with st.spinner("Analyzing fundus..."):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        transform = A.Compose([
            A.Resize(512, 512), 
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        transformed = transform(image=enhanced_np)
        input_tensor = transformed['image'].unsqueeze(0) 
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            max_probs, preds_tensor = torch.max(probs, dim=1)
            max_probs_np = max_probs.squeeze(0).cpu().numpy()
            preds = preds_tensor.squeeze(0).cpu().numpy() 
            
        foreground_pixels = (preds >= 1)
        if np.sum(foreground_pixels) > 0:
            confidence_score = np.mean(max_probs_np[foreground_pixels]) * 100
        else:
            confidence_score = 0.0 
            
        total_pixels = preds.shape[0] * preds.shape[1]
        disc_pixel_count = np.sum(foreground_pixels)
        disc_area_ratio = (disc_pixel_count / total_pixels) * 100
            
    st.markdown("---")
    
    if confidence_score < 85.0 or disc_area_ratio < 0.5:
        st.error("**WARNING: Invalid Image Detected!**")
        st.write(f"- **Low AI Model Confidence**")
        st.write("The AI is not confident in its segmentation, or the detected structure is impossibly small. **Segmentation aborted to prevent misdiagnosis.**")
        st.image(image, use_container_width=True, caption="Original Input")
            
    else:
        st.success(f"**AI Confidence Score:** {confidence_score:.1f}% ")
        
        preds_resized = cv2.resize(preds.astype('uint8'), (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        raw_disc_mask = (preds_resized >= 1) 
        raw_cup_mask = (preds_resized == 1)  
        
        disc_h, disc_contour = clean_mask_and_get_height(raw_disc_mask)
        cup_h, cup_contour = clean_mask_and_get_height(raw_cup_mask)
        
        img_h, img_w = image_np.shape[:2]
        raw_mask_visual = generate_raw_mask_visual(img_h, img_w, disc_contour, cup_contour)
        overlay_visual = overlay_masks(image_np, disc_contour, cup_contour)

        st.subheader("2. Segmentation Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(None , width=400, caption="Optic Nerve Head")
        with col2:
            st.image(None , width=400, caption="predicted Mask")
        with col3:
            st.image(None , width=400, caption="Clinical Overlay (Red: Optic Disc (OD) | Blue: Optic Cup (OC))")
            
        st.markdown("---")
        st.subheader("3. Clinical Assessment")
        
        if disc_h > 0:
            vcdr = cup_h / disc_h
        else:
            vcdr = 0.0
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Model Confidence", value=f"{confidence_score:.1f}%")
        with m_col2:
            st.metric(label="Calculated CDR", value=f"{vcdr:.3f}")
        
        if vcdr <= 0.50:
            st.success("**Diagnosis:** Normal / Low Risk. The vCDR is within healthy limits.")
        elif 0.50 < vcdr <= 0.70:
            st.warning("**Diagnosis:** Glaucoma Suspect. The vCDR is elevated. Clinical correlation recommended.")
        else:
            st.error("**Diagnosis:** High Risk of Glaucoma. The vCDR exceeds 0.70. Immediate specialist review advised.")