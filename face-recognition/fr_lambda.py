import json
import boto3
import torch
import base64
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import io

sqs = boto3.client('sqs')
resp = 'https://sqs.us-east-1.amazonaws.com/728306184817/1234175958-resp-queue'
resnet = InceptionResnetV1(pretrained='vggface2').eval()  # Initialising model outside func to reduce latency

def lambda_handler(event, context):
    if 'Records' not in event or not event['Records']:
        return {
            'body': json.dumps({'error': 'No SQS records found in event'})
        }

    try:
        message = json.loads(event['Records'][0]['body'])
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return {
            'body': json.dumps({'error': f'Failed'})
        }

    required_fields = ['request_id', 'face']
    missing_fields = [field for field in required_fields if field not in message]
    if missing_fields:
        return {
            'body': json.dumps({'error': f'Missing required fields'})
        }

    request_id = message['request_id']
    face_base64 = message['face']

    try:
        face_data = base64.b64decode(face_base64)
        face_pil = Image.open(io.BytesIO(face_data)).convert("RGB")
    except Exception as e:
        return {
            'body': json.dumps({'error': f'Failed to decode face image'})
        }


    try:
        face_numpy = np.array(face_pil, dtype=np.float32) / 255.0
        face_numpy = np.transpose(face_numpy, (2, 0, 1))
        face_tensor = torch.tensor(face_numpy)
    except Exception as e:
        return {
            'body': json.dumps({'error': f'Failed to convert image to tensor'})
        }


    try:
        saved_data = torch.load('resnetV1_video_weights.pt')
        embedding_list = saved_data[0]
        name_list = saved_data[1]
    except Exception as e:
        return {
            'body': json.dumps({'error': f'Failed to load weight'})
        }

    try:
        emb = resnet(face_tensor.unsqueeze(0)).detach()
        dist_list = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]
        idx_min = dist_list.index(min(dist_list))
        result = name_list[idx_min]
    except Exception as e:
        return {
            'body': json.dumps({'error': f'Failed to recognize face'})
        }

    response = {
        'request_id': request_id,
        'result': result
    }
    try:
        sqs.send_message(
            QueueUrl=resp,
            MessageBody=json.dumps(response)
        )
    except Exception as e:
        return {
            'body': json.dumps({'error': f'Failed to send to SQS'})
        }

    return {
        'body': json.dumps({'message': 'Face recognized'})
    }
