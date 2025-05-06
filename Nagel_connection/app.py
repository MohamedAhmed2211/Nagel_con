from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import timm
from torchvision import transforms
from PIL import Image
import io

# تعريف التطبيق
app = FastAPI()

# تحميل النموذج باستخدام الأوزان فقط
model_path = "Nagel2_Resnet2_acc=96,2.pth"  # مسار الأوزان الجديدة
num_classes = 10  # عدد الفئات في النموذج
class_names = [
    'Acral Lentiginous Melanoma', 'Beaus Line', 'Blue Finger', 'Clubbing',
    'Error-Not Nail', 'Healthy Nail', 'Koilonychia', 'Muehrckes Lines',
    'Pitting', 'Terrys Nail'
]

# تحديد الجهاز (CPU أو GPU إذا كان متاحًا)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# إنشاء النموذج بنفس الإعدادات أثناء التدريب
model = timm.create_model("resnet18d", pretrained=False, num_classes=num_classes)

# تحميل الأوزان فقط
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # تفعيل وضع التقييم
    print("✅ Model weights loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model weights: {e}")
    raise HTTPException(status_code=500, detail="❌ Error loading model weights")

# دالة لتحويل الصورة إلى tensor مناسب للنموذج
preprocess = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# دالة التنبؤ
def predict(image: Image.Image):
    image = image.convert("RGB")  # التأكد من أن الصورة بتنسيق RGB
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # تحويل الصورة إلى tensor ونقلها إلى الجهاز المناسب
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # حساب الاحتمالات
    predicted_class_index = probabilities.argmax().item()
    predicted_class_name = class_names[predicted_class_index]
    confidence = probabilities[predicted_class_index].item() * 100
    return {
        "class": predicted_class_name,
        "confidence": confidence,
        "probabilities": {class_names[i]: round(probabilities[i].item() * 100, 2) for i in range(len(class_names))}
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the Nail Disease Classification API! 🚀"}

# **نقطة النهاية لتصنيف الصور**
@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="❌ No file uploaded. Please upload an image file.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"❌ Error opening image: {e}")
    
    result = predict(image)
    return result

#fast\Scripts\Activate.bat
#uvicorn app:app --reload

#ngrok config add-authtoken 2wb5ZC1wcHerJ5oUEvaBAHLpTXr_77UmRcb1Aud9FrHxzpks8
# ngrok http 8000
# https://a4f4-154-182-159-9.ngrok-free.app/


