import json
import boto3
import base64
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import io

sqs = boto3.client('sqs')
req = 'https://sqs.us-east-1.amazonaws.com/728306184817/1234175958-req-queue'
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)  # Initializing at start to reduce latency

def lambda_handler(event, context):
    # Handling error for direct link access
    if 'body' not in event:
        return {
            'body': json.dumps({'error': 'Missing request body'})
        }
    #Handling invalid jsons
    try:
        body = json.loads(event['body'])
    except (json.JSONDecodeError, TypeError):
        return {
            'body': json.dumps({'error': 'Invalid JSON in request body.'})
        }
    
    required_fields = ['content', 'request_id', 'filename']
    missing_fields = [field for field in required_fields if field not in body]
    if missing_fields:
        return {
            'body': json.dumps({'error': f'Missing required fields'})
        }
    
    content = body['content']
    request_id = body['request_id']
    filename = body['filename']
    
    try:
        img_data = base64.b64decode(content)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_array = np.array(img)
        img = Image.fromarray(img_array)
    except Exception as e:
        return {
            'body': json.dumps({'error': f'Failed to decode image'})
        }
    
    face, prob = mtcnn(img, return_prob=True)
    
    if face is not None:
        face_img = face - face.min()
        face_img = face_img / face_img.max()
        face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()
        face_pil = Image.fromarray(face_img, mode="RGB")
        
        buffered = io.BytesIO()
        face_pil.save(buffered, format="JPEG")
        face_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        message = {
            'request_id': request_id,
            'face': face_base64
        }
        try:
            sqs.send_message(
                QueueUrl=req,
                MessageBody=json.dumps(message)
            )
        except Exception as e:
            return {
                'body': json.dumps({'error': f'Failed to send to SQS'})
            }
        
        return {
            'body': json.dumps({'message': 'Face detected and sent to queue'})
        }
    else:
        return {
            'body': json.dumps({'message': 'No face detected'})
        }
