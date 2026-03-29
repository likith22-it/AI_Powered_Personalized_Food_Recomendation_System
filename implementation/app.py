import base64
import os

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import timm
from flask import Flask, redirect, render_template, request, session, url_for
import functools
import hashlib
import json
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-please")

USER_DB_PATH = os.path.join(os.path.dirname(__file__), "users.json")
NUTRITION_DATA_PATH = os.path.join(os.path.dirname(__file__), "food101.json")

# Load food nutrition facts (macro + micro nutrient data)
try:
    with open(NUTRITION_DATA_PATH, "r", encoding="utf-8") as f:
        NUTRITION_DATA = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    NUTRITION_DATA = {}

# Simple file-backed user store
def _load_users():
    try:
        with open(USER_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_users(users):
    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return view(**kwargs)

    return wrapped_view

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "models")
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", os.path.join(MODEL_DIR, "best.pt"))
VIT_WEIGHTS = os.getenv("VIT_WEIGHTS", os.path.join(MODEL_DIR, "vit_tiny_food_models.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model loading (run once at startup)
# ---------------------------------------------------------------------------
print("Loading models...")

# 1) YOLOv8 detector
print(f"Loading YOLO weights from: {YOLO_WEIGHTS}")
yolo_model = YOLO(YOLO_WEIGHTS)

# 2) ViT classifier (101 food classes)
NUM_CLASSES = 101
print(f"Loading ViT classifier weights from: {VIT_WEIGHTS}")
vit_model = timm.create_model(
    "vit_tiny_patch16_224.augreg_in21k", pretrained=False, num_classes=NUM_CLASSES
)

state_dict = torch.load(VIT_WEIGHTS, map_location=device)
# Some checkpoints include a "timm_model." prefix.
new_state_dict = {k.replace("timm_model.", ""): v for k, v in state_dict.items()}
vit_model.load_state_dict(new_state_dict)
vit_model.to(device)
vit_model.eval()

# 3) MiDaS depth model
print("Loading MiDaS depth model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform_depth = midas_transforms.small_transform

# ---------------------------------------------------------------------------
# Diet recommendation model (used to decide what foods to avoid/limit)
# ---------------------------------------------------------------------------
RECOMMENDATION_DIR = os.getenv("RECOMMENDATION_DIR", "RECOMENDATION_MODULE")
DIET_MODEL_PATH = os.path.join(RECOMMENDATION_DIR, "diet_model.pkl")
DIET_ENCODER_PATH = os.path.join(RECOMMENDATION_DIR, "diet_encoder.pkl")
DIET_FEATURES_PATH = os.path.join(RECOMMENDATION_DIR, "model_features.pkl")

print(f"Loading diet recommendation model from: {DIET_MODEL_PATH}")
model = joblib.load(DIET_MODEL_PATH)
encoder = joblib.load(DIET_ENCODER_PATH)
features = joblib.load(DIET_FEATURES_PATH)

# Nutrient data (per 100g) used for micro/macro nutrient calculation.
FOOD_NUTRITION_PATH = os.path.join(os.path.dirname(__file__), "food101.json")
try:
    with open(FOOD_NUTRITION_PATH, "r", encoding="utf-8") as f:
        food_nutrition = json.load(f)
except Exception:
    food_nutrition = {}

# Food categories used by the recommendation system
food_category_map = {
    "apple_pie": "high_sugar",
    "baby_back_ribs": "high_fat",
    "baklava": "high_sugar",
    "beef_carpaccio": "high_protein",
    "beef_tartare": "high_protein",
    "beet_salad": "healthy_food",
    "beignets": "high_sugar",
    "bibimbap": "mixed_meal",
    "bread_pudding": "high_sugar",
    "breakfast_burrito": "mixed_meal",
    "bruschetta": "refined_carbs",
    "caesar_salad": "healthy_food",
    "cannoli": "high_sugar",
    "caprese_salad": "healthy_food",
    "carrot_cake": "high_sugar",
    "ceviche": "high_protein",
    "cheesecake": "high_sugar",
    "cheese_plate": "high_fat",
    "chicken_curry": "high_protein",
    "chicken_quesadilla": "mixed_meal",
    "chicken_wings": "fried_food",
    "chocolate_cake": "high_sugar",
    "chocolate_mousse": "high_sugar",
    "churros": "high_sugar",
    "clam_chowder": "high_sodium",
    "club_sandwich": "refined_carbs",
    "crab_cakes": "fried_food",
    "creme_brulee": "high_sugar",
    "croque_madame": "refined_carbs",
    "cup_cakes": "high_sugar",
    "deviled_eggs": "high_protein",
    "donuts": "high_sugar",
    "dumplings": "refined_carbs",
    "edamame": "healthy_food",
    "eggs_benedict": "high_protein",
    "escargots": "high_protein",
    "falafel": "fried_food",
    "filet_mignon": "high_protein",
    "fish_and_chips": "fried_food",
    "foie_gras": "high_fat",
    "french_fries": "fried_food",
    "french_onion_soup": "high_sodium",
    "french_toast": "refined_carbs",
    "fried_calamari": "fried_food",
    "fried_rice": "refined_carbs",
    "frozen_yogurt": "high_sugar",
    "garlic_bread": "refined_carbs",
    "gnocchi": "refined_carbs",
    "greek_salad": "healthy_food",
    "grilled_cheese": "high_fat",
    "grilled_cheese_sandwich": "high_fat",
    "grilled_salmon": "high_protein",
    "guacamole": "healthy_food",
    "gyoza": "fried_food",
    "hamburger": "refined_carbs",
    "hot_and_sour_soup": "high_sodium",
    "hot_dog": "high_sodium",
    "huevos_rancheros": "mixed_meal",
    "hummus": "healthy_food",
    "ice_cream": "high_sugar",
    "lasagna": "refined_carbs",
    "lobster_bisque": "high_fat",
    "lobster_roll_sandwich": "refined_carbs",
    "macaroni_and_cheese": "refined_carbs",
    "macarons": "high_sugar",
    "miso_soup": "high_sodium",
    "mussels": "high_protein",
    "nachos": "high_fat",
    "omelette": "high_protein",
    "onion_rings": "fried_food",
    "oysters": "high_protein",
    "pad_thai": "refined_carbs",
    "paella": "mixed_meal",
    "pancakes": "refined_carbs",
    "panna_cotta": "high_sugar",
    "peking_duck": "high_fat",
    "pho": "mixed_meal",
    "pizza": "refined_carbs",
    "pork_chop": "high_protein",
    "poutine": "high_fat",
    "prime_rib": "high_fat",
    "pulled_pork_sandwich": "refined_carbs",
    "ramen": "refined_carbs",
    "ravioli": "refined_carbs",
    "red_velvet_cake": "high_sugar",
    "risotto": "refined_carbs",
    "samosa": "fried_food",
    "sashimi": "high_protein",
    "scallops": "high_protein",
    "seaweed_salad": "healthy_food",
    "shrimp_and_grits": "mixed_meal",
    "spaghetti_bolognese": "refined_carbs",
    "spaghetti_carbonara": "refined_carbs",
    "spring_rolls": "fried_food",
    "steak": "high_protein",
    "strawberry_shortcake": "high_sugar",
    "sushi": "mixed_meal",
    "tacos": "mixed_meal",
    "takoyaki": "fried_food",
    "tiramisu": "high_sugar",
    "tuna_tartare": "high_protein",
    "waffles": "refined_carbs",
}

diet_rules= {
    "Low_Carb": {
        "avoid": ["refined_carbs", "high_sugar"],
        "limit": ["fried_food", "high_fat", "mixed_meal"],
        "prefer": ["high_protein", "healthy_food"],
    },
    "Low_Sodium": {
        "avoid": ["high_sodium", "fried_food"],
        "limit": ["high_fat", "refined_carbs", "mixed_meal"],
        "prefer": ["healthy_food", "high_protein"],
    },
    "Balanced": {
        "avoid": ["high_sugar"],
        "limit": ["fried_food", "high_fat"],
        "prefer": ["healthy_food", "high_protein", "mixed_meal", "refined_carbs"],
    },
}


def diet_food_recommendation(age, gender, weight, height, disease, severity, imbalance_score, detected_foods):
    """Return diet recommendation and which detected foods to avoid/limit.

    This function uses the trained diet model to predict a recommended diet type and
    compares detected foods against diet rules.
    """

    # calculate BMI
    bmi = weight / ((height / 100) ** 2)

    # encode categorical inputs
    gender_enc = 1 if gender.lower() == "male" else 0

    disease_map = {"none": 0, "diabetes": 1, "hypertension": 2, "obesity": 3}
    severity_map = {"mild": 0, "moderate": 1, "severe": 2}

    disease_enc = disease_map.get(disease.lower(), 0)
    severity_enc = severity_map.get(severity.lower(), 0)

    # create dataframe for prediction
    sample = pd.DataFrame(
        [[age, gender_enc, weight, height, bmi, disease_enc, severity_enc, imbalance_score]],
        columns=features,
    )

    # predict diet and map back to label
    pred = model.predict(sample)
    diet = encoder.inverse_transform(pred)[0]

    avoid_foods = []
    limit_foods = []
    safe_foods = []

    for food in detected_foods:
        category = food_category_map.get(food)
        if category is None:
            safe_foods.append(food)
            continue

        if category in diet_rules.get(diet, {}).get("avoid", []):
            avoid_foods.append(food)
        elif category in diet_rules.get(diet, {}).get("limit", []):
            limit_foods.append(food)
        else:
            safe_foods.append(food)

    # Suggest foods based on diet preferences and the categories of detected foods.
    # If the predicted diet is itself a food category (e.g. "refined_carbs"), suggest from that category.
    category_values = set(food_category_map.values())
    if diet in category_values:
        prefer_categories = {diet}
    else:
        prefer_categories = set(diet_rules.get(diet, {}).get("prefer", []))

    # Include detected food categories so suggestions vary depending on what was seen.
    detected_categories = {
        food_category_map.get(f)
        for f in detected_foods
        if f in food_category_map
    }

    suggest_categories = prefer_categories | detected_categories

    suggested_foods = []
    for f, cat in food_category_map.items():
        # Skip beef-related foods for cultural/religious reasons.
        if "beef" in f:
            continue
        if cat in suggest_categories and f not in detected_foods:
            suggested_foods.append(f)

    # Limit suggestions to a reasonable number for display
    suggested_foods = sorted(list(dict.fromkeys(suggested_foods)))[:10]

    return {
        "Recommended_Diet": diet,
        "Detected_Foods": detected_foods,
        "Avoid_Foods": avoid_foods,
        "Limit_Foods": limit_foods,
        "Safe_Foods": safe_foods,
        "Suggested_Foods": suggested_foods,
        # For alerting UI logic if the user is eating something they should avoid.
        "Has_Avoid_Alert": len(avoid_foods) > 0,
    }

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheese_plate",
    "cheesecake",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "escargots",
    "falafel",
    "filet_mignon",
    "fish_and_chips",
    "foie_gras",
    "french_fries",
    "french_onion_soup",
    "french_toast",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "huevos_rancheros",
    "hummus",
    "ice_cream",
    "lasagna",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "macaroni_and_cheese",
    "macarons",
    "miso_soup",
    "mussels",
    "nachos",
    "omelette",
    "onion_rings",
    "oysters",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "peking_duck",
    "pho",
    "pizza",
    "pork_chop",
    "poutine",
    "prime_rib",
    "pulled_pork_sandwich",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed_salad",
    "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna_tartare",
    "waffles",
]

FOOD_DENSITY = {
    "apple_pie": 0.75,
    "baby_back_ribs": 1.05,
    "baklava": 0.75,
    "beef_carpaccio": 1.05,
    "beef_tartare": 1.05,
    "beet_salad": 0.95,
    "beignets": 0.60,
    "bibimbap": 0.95,
    "bread_pudding": 0.75,
    "breakfast_burrito": 0.95,
    "bruschetta": 0.85,
    "caesar_salad": 0.90,
    "cannoli": 0.70,
    "caprese_salad": 0.90,
    "carrot_cake": 0.70,
    "ceviche": 1.00,
    "cheese_plate": 1.05,
    "cheesecake": 0.80,
    "chicken_curry": 1.00,
    "chicken_quesadilla": 0.95,
    "chicken_wings": 1.05,
    "chocolate_cake": 0.70,
    "chocolate_mousse": 0.50,
    "churros": 0.65,
    "clam_chowder": 1.00,
    "club_sandwich": 0.95,
    "crab_cakes": 1.00,
    "creme_brulee": 1.00,
    "croque_madame": 1.05,
    "cup_cakes": 0.65,
    "deviled_eggs": 1.05,
    "donuts": 0.60,
    "dumplings": 0.95,
    "edamame": 1.00,
    "eggs_benedict": 1.05,
    "escargots": 1.05,
    "falafel": 0.90,
    "filet_mignon": 1.05,
    "fish_and_chips": 1.00,
    "foie_gras": 1.05,
    "french_fries": 0.60,
    "french_onion_soup": 1.00,
    "french_toast": 0.80,
    "fried_calamari": 0.90,
    "fried_rice": 0.95,
    "frozen_yogurt": 0.60,
    "garlic_bread": 0.70,
    "gnocchi": 0.95,
    "greek_salad": 0.90,
    "grilled_cheese_sandwich": 0.90,
    "grilled_salmon": 1.05,
    "guacamole": 0.95,
    "gyoza": 0.95,
    "hamburger": 0.95,
    "hot_and_sour_soup": 1.00,
    "hot_dog": 0.95,
    "huevos_rancheros": 1.05,
    "hummus": 0.95,
    "ice_cream": 0.60,
    "lasagna": 0.95,
    "lobster_bisque": 1.00,
    "lobster_roll_sandwich": 0.95,
    "macaroni_and_cheese": 0.90,
    "macarons": 0.55,
    "miso_soup": 1.00,
    "mussels": 1.05,
    "nachos": 0.70,
    "omelette": 1.05,
    "onion_rings": 0.65,
    "oysters": 1.05,
    "pad_thai": 0.95,
    "paella": 0.95,
    "pancakes": 0.70,
    "panna_cotta": 0.95,
    "peking_duck": 1.05,
    "pho": 1.00,
    "pizza": 0.85,
    "pork_chop": 1.05,
    "poutine": 0.95,
    "prime_rib": 1.05,
    "pulled_pork_sandwich": 0.95,
    "ramen": 1.00,
    "ravioli": 0.95,
    "red_velvet_cake": 0.70,
    "risotto": 0.95,
    "samosa": 0.85,
    "sashimi": 1.05,
    "scallops": 1.05,
    "seaweed_salad": 0.90,
    "shrimp_and_grits": 0.95,
    "spaghetti_bolognese": 0.95,
    "spaghetti_carbonara": 0.95,
    "spring_rolls": 0.85,
    "steak": 1.05,
    "strawberry_shortcake": 0.70,
    "sushi": 0.90,
    "tacos": 0.90,
    "takoyaki": 0.85,
    "tiramisu": 0.75,
    "tuna_tartare": 1.05,
    "waffles": 0.70,
}

SCALE = 1e-6


vit_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def classify_food(crop_rgb: np.ndarray) -> str:
    """Classify a cropped food patch using the ViT classifier."""

    pil = Image.fromarray(crop_rgb)
    input_tensor = vit_transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = vit_model(input_tensor)
        pred = torch.argmax(outputs, dim=1).item()
    return CLASS_NAMES[pred]


def compute_depth_map(image_rgb: np.ndarray) -> np.ndarray:
    """Compute a depth map from an RGB image using MiDaS."""

    input_batch = transform_depth(image_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
    depth_map = prediction.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (image_rgb.shape[1], image_rgb.shape[0]))
    return depth_map


@app.route("/", methods=["GET"])
def index():
    # Redirect users to log in if not authenticated.
    if not session.get("user"):
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            error = "Email and password are required."
        else:
            users = _load_users()
            if email in users:
                error = "An account with that email already exists."
            else:
                users[email] = {"password": _hash_password(password)}
                _save_users(users)
                # Don't auto-login after signup; require explicit login.
                return redirect(url_for("login", msg="Account created. Please log in."))

    return render_template("signup.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    message = request.args.get("msg")
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        users = _load_users()
        stored = users.get(email)
        if not stored or stored.get("password") != _hash_password(password):
            error = "Invalid email or password."
        else:
            session["user"] = email
            return redirect(url_for("index"))

    return render_template("login.html", error=error, message=message)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    data = file.read()
    np_img = np.frombuffer(data, np.uint8)
    image_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return "Unable to decode image", 400

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Depth estimation
    depth_map = compute_depth_map(image_rgb)

    # Object detection
    results = yolo_model(image_bgr, conf=0.2)

    # Use a smaller scale when multiple foods are detected (more objects => smaller per-object estimate)
    num_detections = 0
    if len(results) > 0 and hasattr(results[0], "boxes"):
        num_detections = len(results[0].boxes)
    SCALE = 1e-6 if num_detections == 1 else 1e-7

    detections = []
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_rgb.shape[1] - 1, x2)
            y2 = min(image_rgb.shape[0] - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop_rgb = image_rgb[y1:y2, x1:x2]
            label = classify_food(crop_rgb)

            depth_region = depth_map[y1:y2, x1:x2]
            avg_depth = float(depth_region.mean())

            area = (x2 - x1) * (y2 - y1)
            volume = area * avg_depth
            density = FOOD_DENSITY.get(label, 1.0)
            grams = volume * density * SCALE

            nutrition = {}
            if label in NUTRITION_DATA and grams > 0:
                scale = grams / 100.0
                nutrition = {
                    k: round(v * scale, 1)
                    for k, v in NUTRITION_DATA[label].items()
                    if isinstance(v, (int, float))
                }

            detections.append(
                {
                    "label": label,
                    "confidence": float(conf),
                    "box": [x1, y1, x2, y2],
                    "avg_depth": avg_depth,
                    "grams": grams,
                    "nutrition": nutrition,
                }
            )

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image_bgr,
                f"{label} {conf:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    # Gather diet recommendation inputs (from the form)
    try:
        age = int(request.form.get("age", 30))
        weight = float(request.form.get("weight", 70))
        height = float(request.form.get("height", 170))
        gender = request.form.get("gender", "male")
        disease = request.form.get("disease", "none")
        severity = request.form.get("severity", "mild")
        imbalance_score = float(request.form.get("imbalance_score", 0.0))
    except ValueError:
        # If parsing fails, fall back to defaults
        age = 30
        weight = 70.0
        height = 170.0
        gender = "male"
        disease = "none"
        severity = "mild"
        imbalance_score = 0.0

    detected_labels = [d["label"] for d in detections]
    diet_result = diet_food_recommendation(
        age, gender, weight, height, disease, severity, imbalance_score, detected_labels
    )

    # Encode annotated image for display
    _, buffer = cv2.imencode(".jpg", image_bgr)
    image_b64 = base64.b64encode(buffer).decode("utf-8")

    return render_template(
        "result.html",
        detections=detections,
        image_data=image_b64,
        diet_result=diet_result,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
