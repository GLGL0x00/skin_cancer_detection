import json 
import torch
import base64
import csv
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO, StringIO
from catboost import CatBoostClassifier

# Paths to the models (relative to the Lambda deployment)
IMAGE_MODEL_PATH = 'efficientnet_best.pth'
METADATA_MODEL_PATH = 'catboost_model.cbm'

def load_models():
    # Load image model (PyTorch)
    image_model = torch.load(IMAGE_MODEL_PATH, map_location=torch.device('cpu'))  # Loading the model to CPU
    image_model.eval()  # Set to evaluation mode

    # Load metadata model (CatBoost)
    metadata_model = CatBoostClassifier()
    metadata_model.load_model(METADATA_MODEL_PATH)

    return image_model, metadata_model


def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Define image transformations
    preprocess_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_image = preprocess_transform(image)
    return processed_image

import csv
from io import StringIO
import json

def preprocess_metadata(metadata_csv_bytes):
    # Decode the CSV bytes
    csv_string = metadata_csv_bytes.decode('utf-8')
    csv_reader = csv.reader(StringIO(csv_string), delimiter=',')
    
    processed_metadata = []
    categorical_features = ['sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location_simple']
    
    # Skip the header row
    for row in csv_reader:
        if csv_reader.line_num == 1:
            continue
        
        # Extract relevant metadata fields 
        isic_id = row[0]  
        target = row[1]  # Target is usually part of the training data
        age_approx = float(row[2]) if row[2] else None  # Convert to float if available
        sex = row[3]
        anatom_site_general = row[4]
        clin_size_long_diam_mm = float(row[5]) if row[5] else None  # Convert to float if available
        tbp_tile_type = row[7]
        tbp_lv_A = float(row[8]) if row[8] else None
        tbp_lv_Aext = float(row[9]) if row[9] else None
        tbp_lv_B = float(row[10]) if row[10] else None
        tbp_lv_Bext = float(row[11]) if row[11] else None
        tbp_lv_C = float(row[12]) if row[12] else None
        tbp_lv_Cext = float(row[13]) if row[13] else None
        tbp_lv_H = float(row[14]) if row[14] else None
        tbp_lv_Hext = float(row[15]) if row[15] else None
        tbp_lv_L = float(row[16]) if row[16] else None
        tbp_lv_Lext = float(row[17]) if row[17] else None
        tbp_lv_areaMM2 = float(row[18]) if row[18] else None
        tbp_lv_area_perim_ratio = float(row[19]) if row[19] else None
        tbp_lv_color_std_mean = float(row[20]) if row[20] else None
        tbp_lv_deltaA = float(row[21]) if row[21] else None
        tbp_lv_deltaB = float(row[22]) if row[22] else None
        tbp_lv_deltaL = float(row[23]) if row[23] else None
        tbp_lv_deltaLB = float(row[24]) if row[24] else None
        tbp_lv_deltaLBnorm = float(row[25]) if row[25] else None
        tbp_lv_eccentricity = float(row[26]) if row[26] else None
        tbp_lv_location_simple = row[28]
        tbp_lv_minorAxisMM = float(row[29]) if row[29] else None
        tbp_lv_nevi_confidence = float(row[30]) if row[30] else None
        tbp_lv_norm_border = float(row[31]) if row[31] else None
        tbp_lv_norm_color = float(row[32]) if row[32] else None
        tbp_lv_perimeterMM = float(row[33]) if row[33] else None
        tbp_lv_radial_color_std_max = float(row[34]) if row[34] else None
        tbp_lv_stdL = float(row[35]) if row[35] else None
        tbp_lv_stdLExt = float(row[36]) if row[36] else None
        tbp_lv_symm_2axis = float(row[37]) if row[37] else None
        tbp_lv_symm_2axis_angle = float(row[38]) if row[38] else None
        tbp_lv_x = float(row[39]) if row[39] else None
        tbp_lv_y = float(row[40]) if row[40] else None
        tbp_lv_z = float(row[41]) if row[41] else None

        # Append processed metadata for the CatBoost model
        processed_metadata.append([
            isic_id, target, age_approx, sex, anatom_site_general, clin_size_long_diam_mm, tbp_tile_type, tbp_lv_A,
            tbp_lv_Aext, tbp_lv_B, tbp_lv_Bext, tbp_lv_C, tbp_lv_Cext, tbp_lv_H, tbp_lv_Hext, tbp_lv_L, tbp_lv_Lext,
            tbp_lv_areaMM2, tbp_lv_area_perim_ratio, tbp_lv_color_std_mean, tbp_lv_deltaA, tbp_lv_deltaB, tbp_lv_deltaL,
            tbp_lv_deltaLB, tbp_lv_deltaLBnorm, tbp_lv_eccentricity, tbp_lv_location_simple, tbp_lv_minorAxisMM,
            tbp_lv_nevi_confidence, tbp_lv_norm_border, tbp_lv_norm_color, tbp_lv_perimeterMM, tbp_lv_radial_color_std_max,
            tbp_lv_stdL, tbp_lv_stdLExt, tbp_lv_symm_2axis, tbp_lv_symm_2axis_angle, tbp_lv_x, tbp_lv_y, tbp_lv_z
        ])

    return processed_metadata[0], categorical_features
  # Returning the first row of processed metadata

def lambda_handler(event, context):
    try:
        # Parse the image file and metadata CSV from the form data
        image_data = base64.b64decode(event['body']['image'])  # Decoding base64-encoded image
        metadata_csv = base64.b64decode(event['body']['metadata_csv'])  # Decoding base64-encoded CSV file

        # Load models
        image_model, metadata_model = load_models()

        # Preprocess the image and metadata
        processed_image = preprocess_image(image_data)
        processed_metadata = preprocess_metadata(metadata_csv)

        # Run inference on the image model
        with torch.no_grad():
            image_prediction = image_model(processed_image.unsqueeze(0))  # Add batch dimension
            image_prediction = image_prediction.item()  # Convert to scalar

        # Run inference on the metadata model
        metadata_prediction = metadata_model.predict([processed_metadata])[0]  # Predict returns an array, get the first value

        # Combine predictions (weighted averaging)
        combined_prediction = 0.6 * image_prediction + 0.4 * metadata_prediction

        # Return the prediction result
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': str(combined_prediction)
            })
        }
    
    except Exception as e:
        # Handle any exceptions during processing
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }